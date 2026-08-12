# %% [markdown]
# # Experimento 06.1 — Editable Memory
# **Gerenciamento de contexto e memória editável em agentes de IA**
#
# `LLM != agente`. Uma LLM processa contexto; um agente combina LLM, estado,
# memória, ferramentas, regras e controle de execução.

# %% [markdown]
# ## O problema da context window
# A janela recebe system prompt, core memory, mensagens, tool calls, resultados e
# documentos. Como é limitada, a arquitetura decide o que manter, remover,
# compactar, armazenar externamente e recuperar depois.

# %% [markdown]
# ## Analogia com sistema operacional
# `RAM ↔ memória virtual ↔ disco` inspira a comparação conceitual
# `Context Window ↔ Core Memory ↔ Archival Memory`. Os mecanismos não são
# tecnicamente idênticos; a analogia ajuda a pensar em recursos limitados.

# %% [markdown]
# ## Instalação
# ```bash
# pip install -r requirements.txt
# ```
# Opcional no notebook: `# %pip install -r requirements.txt`.
# Para Letta: `# %pip install letta-client`.

# %%
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

def locate_project_dir() -> Path:
    if "__file__" in globals(): return Path(__file__).resolve().parent
    current = Path.cwd().resolve()
    for candidate in [current, current / "experiments" / "06-gerenciamento-contexto", *[p / "experiments" / "06-gerenciamento-contexto" for p in current.parents]]:
        if (candidate / "data" / "memory_seed.json").is_file(): return candidate
    return current

PROJECT_DIR = locate_project_dir()
REPO_ROOT = next((path for path in (PROJECT_DIR, *PROJECT_DIR.parents) if (path / "experiments").is_dir()), PROJECT_DIR)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(PROJECT_DIR / ".env", override=False)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip().removeprefix("groq/")
LETTA_API_KEY = os.getenv("LETTA_API_KEY", "").strip()
LETTA_MODEL = os.getenv("LETTA_MODEL", f"groq/{GROQ_MODEL}").strip()
RUN_LIVE_DEMOS = os.getenv("RUN_LIVE_DEMOS", "false").lower() == "true"
GROQ_READY = bool(GROQ_API_KEY and GROQ_API_KEY != "cole_sua_chave_aqui")
if __name__ == "__main__":
    print(f"Groq configurada: {'sim' if GROQ_READY else 'não'}")
    print(f"Modelo: {GROQ_MODEL}")
    print(f"Letta configurado: {'sim' if LETTA_API_KEY else 'não (opcional)'}")

# %%
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# %%
class MemoryBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value: str
    limit: int = Field(gt=0)
    read_only: bool = False

    @model_validator(mode="after")
    def value_fits(self) -> "MemoryBlock":
        if len(self.value) > self.limit:
            raise ValueError(f"MemoryBlock '{self.label}' excede o limite: {len(self.value)}/{self.limit} caracteres.")
        return self

# %%
class EditableMemoryManager:
    def __init__(self, blocks: list[MemoryBlock]):
        self._blocks = {block.label: block.model_copy(deep=True) for block in blocks}
        if len(self._blocks) != len(blocks): raise ValueError("Labels de memória devem ser únicos.")

    def get_block(self, label: str) -> MemoryBlock:
        if label not in self._blocks: raise KeyError(f"MemoryBlock inexistente: {label}")
        return self._blocks[label].model_copy(deep=True)

    def list_blocks(self) -> list[MemoryBlock]: return [b.model_copy(deep=True) for b in self._blocks.values()]

    def _write(self, label: str, value: str) -> MemoryBlock:
        block = self.get_block(label)
        if block.read_only: raise PermissionError(f"MemoryBlock read-only: {label}")
        updated = block.model_copy(update={"value": value})
        updated = MemoryBlock.model_validate(updated.model_dump())
        self._blocks[label] = updated
        return updated.model_copy(deep=True)

    def update_block(self, label: str, new_value: str) -> MemoryBlock: return self._write(label, new_value.strip())
    def append_to_block(self, label: str, content: str) -> MemoryBlock:
        block = self.get_block(label); addition = content.strip()
        if not addition or addition in block.value: return block
        return self._write(label, f"{block.value.rstrip()}\n{addition}")
    def replace_in_block(self, label: str, old: str, new: str) -> MemoryBlock:
        block = self.get_block(label)
        if old not in block.value: raise ValueError(f"Texto a substituir não encontrado no bloco '{label}'.")
        return self._write(label, block.value.replace(old, new))
    def delete_from_block(self, label: str, text: str) -> MemoryBlock:
        block = self.get_block(label)
        if text not in block.value: raise ValueError(f"Texto a remover não encontrado no bloco '{label}'.")
        return self._write(label, block.value.replace(text, "").strip())
    def render_context(self) -> str:
        return "\n\n".join(f"[{b.label.upper()}]\nDescrição: {b.description}\nConteúdo: {b.value}" for b in self._blocks.values())
    def to_dict(self) -> dict[str, Any]: return {b.label: b.model_dump(exclude={"label"}) for b in self._blocks.values()}

# %%
def load_seed(path: Path | None = None) -> EditableMemoryManager:
    seed_path = path or PROJECT_DIR / "data" / "memory_seed.json"
    try: data = json.loads(seed_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e: raise FileNotFoundError(f"Memory seed inexistente: {seed_path}") from e
    except json.JSONDecodeError as e: raise ValueError(f"JSON inválido em {seed_path}: {e}") from e
    blocks = [MemoryBlock(label=label, **content) for label, content in data.items()]
    return EditableMemoryManager(blocks)

memory_manager = load_seed()
if __name__ == "__main__":
    for label in ("persona", "human", "project"): print(f"\n{label.upper()}\n{memory_manager.get_block(label).value}")

# %%
def display_memory(manager: EditableMemoryManager) -> pd.DataFrame:
    frame = pd.DataFrame([{"label": b.label, "description": b.description, "chars_used": len(b.value), "limit": b.limit, "read_only": b.read_only, "value": b.value} for b in manager.list_blocks()])
    print(frame.to_string(index=False)); return frame

# %% [markdown]
# ## Core Memory
# Informação importante o suficiente para permanecer disponível no contexto ativo:
# persona, preferências, objetivo atual e regras essenciais.
# `LLM Context = System Prompt + Core Memory + Recent Messages`.

# %%
SYSTEM_PROMPT = """Você é o ContextMemoryAgent, um agente com memória editável.
Antes de responder: analise a mensagem; identifique fatos duráveis e reutilizáveis;
edite memória somente quando necessário; não salve trivialidades; não duplique fatos;
substitua preferências que mudaram em vez de manter contradições; respeite limites e
blocos read-only. Toda alteração deve ocorrer por uma memory tool. Depois responda
usando a memória atual. Nunca revele chaves nem alegue ter alterado memória sem tool call."""

# %%
def memory_get(manager: EditableMemoryManager, label: str) -> dict[str, Any]: return manager.get_block(label).model_dump()
def memory_update(manager: EditableMemoryManager, label: str, new_value: str) -> dict[str, Any]: return manager.update_block(label, new_value).model_dump()
def memory_append(manager: EditableMemoryManager, label: str, content: str) -> dict[str, Any]: return manager.append_to_block(label, content).model_dump()
def memory_replace(manager: EditableMemoryManager, label: str, old: str, new: str) -> dict[str, Any]: return manager.replace_in_block(label, old, new).model_dump()
def memory_delete(manager: EditableMemoryManager, label: str, text: str) -> dict[str, Any]: return manager.delete_from_block(label, text).model_dump()

MEMORY_TOOLS = [
    {"type":"function","name":"memory_get","description":"Lê um bloco de memória.","parameters":{"type":"object","properties":{"label":{"type":"string"}},"required":["label"],"additionalProperties":False},"strict":True},
    {"type":"function","name":"memory_update","description":"Substitui todo o valor de um bloco editável.","parameters":{"type":"object","properties":{"label":{"type":"string"},"new_value":{"type":"string"}},"required":["label","new_value"],"additionalProperties":False},"strict":True},
    {"type":"function","name":"memory_append","description":"Acrescenta informação nova e não redundante.","parameters":{"type":"object","properties":{"label":{"type":"string"},"content":{"type":"string"}},"required":["label","content"],"additionalProperties":False},"strict":True},
    {"type":"function","name":"memory_replace","description":"Substitui texto desatualizado por informação nova.","parameters":{"type":"object","properties":{"label":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["label","old","new"],"additionalProperties":False},"strict":True},
    {"type":"function","name":"memory_delete","description":"Remove texto desatualizado de um bloco.","parameters":{"type":"object","properties":{"label":{"type":"string"},"text":{"type":"string"}},"required":["label","text"],"additionalProperties":False},"strict":True},
]
CHAT_MEMORY_TOOLS = [
    {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
    for tool in MEMORY_TOOLS
]

# %%
MAX_TOOL_STEPS = 5
@dataclass
class TurnTrace:
    message: str; memory_before: str; memory_after: str = ""; context: str = ""; response: str = ""; tool_calls: list[dict[str, Any]] = field(default_factory=list)

class ContextMemoryAgent:
    def __init__(self, manager: EditableMemoryManager, client: Groq | Any | None = None, model: str = GROQ_MODEL):
        self.memory = manager; self.client = client; self.model = model; self.recent_messages: list[dict[str,str]] = []
    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        functions = {"memory_get":memory_get,"memory_update":memory_update,"memory_append":memory_append,"memory_replace":memory_replace,"memory_delete":memory_delete}
        if name not in functions: raise ValueError(f"Memory tool desconhecida: {name}")
        return functions[name](self.memory, **args)
    def chat(self, message: str) -> TurnTrace:
        if not message.strip(): raise ValueError("Mensagem vazia.")
        if self.client is None: raise RuntimeError("GROQ_API_KEY ausente: crie o cliente apenas após configurar .env.")
        before = self.memory.render_context(); recent = "\n".join(f"{m['role']}: {m['content']}" for m in self.recent_messages[-6:])
        instructions = f"{SYSTEM_PROMPT}\n\nCORE MEMORY:\n{before}\n\nMENSAGENS RECENTES:\n{recent or '(nenhuma)'}"
        messages: list[Any] = [{"role":"system","content":instructions},{"role":"user","content":message}]; trace = TurnTrace(message=message,memory_before=before,context=instructions)
        for step in range(MAX_TOOL_STEPS):
            tool_choice = "none" if step == MAX_TOOL_STEPS - 1 else "auto"
            try: response = self.client.chat.completions.create(model=self.model,messages=messages,tools=CHAT_MEMORY_TOOLS,tool_choice=tool_choice)
            except Exception as e: raise RuntimeError(f"Erro da LLM Groq ({type(e).__name__}): {e}") from e
            assistant_message = response.choices[0].message
            calls = list(assistant_message.tool_calls or [])
            if not calls:
                text = (assistant_message.content or "").strip()
                if not text: raise ValueError("Resposta vazia da LLM.")
                trace.response=text; trace.memory_after=self.memory.render_context(); self.recent_messages.extend([{"role":"user","content":message},{"role":"assistant","content":text}]); return trace
            messages.append(assistant_message.model_dump(exclude_none=True))
            for call in calls:
                args: dict[str, Any] = {}
                try: args=json.loads(call.function.arguments); result=self._dispatch(call.function.name,args)
                except Exception as e: result={"error":f"{type(e).__name__}: {e}"}
                trace.tool_calls.append({"name":call.function.name,"arguments":args,"result":result})
                messages.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(result,ensure_ascii=False)})
            instructions = f"{SYSTEM_PROMPT}\n\nCORE MEMORY ATUALIZADA:\n{self.memory.render_context()}\n\nResponda após concluir as edições necessárias."
            messages[0] = {"role":"system","content":instructions}
        raise RuntimeError(f"Loop de tool calling excedido: máximo {MAX_TOOL_STEPS} passos.")

def create_groq_agent(manager: EditableMemoryManager | None = None) -> ContextMemoryAgent:
    if not GROQ_READY: raise RuntimeError("GROQ_API_KEY ausente ou com valor de exemplo.")
    client = Groq(api_key=GROQ_API_KEY)
    return ContextMemoryAgent(manager or load_seed(), client, GROQ_MODEL)

# %%
FIRST_MESSAGE = """Meu nome é Rafael. Estou estudando inteligência artificial e atualmente estou focado em LangGraph, RAG, A2A e Knowledge Graph. Prefiro explicações simples seguidas de exemplos práticos."""
live_agent: ContextMemoryAgent | None = None
if __name__ == "__main__" and RUN_LIVE_DEMOS and GROQ_READY:
    live_agent=create_groq_agent(); print("MEMÓRIA ANTES"); display_memory(live_agent.memory); print(live_agent.chat(FIRST_MESSAGE).response); print("MEMÓRIA DEPOIS"); display_memory(live_agent.memory)

# %%
if __name__ == "__main__" and live_agent is not None:
    trace=live_agent.chat("Explique o que é um vector database."); print("Informações utilizadas da memória:"); print(live_agent.memory.get_block("human").value); print(trace.response)

# %%
PREFERENCE_CHANGE = "Mudei de preferência. Quando eu perguntar algo técnico, primeiro dê o exemplo prático e depois explique o conceito."
if __name__ == "__main__" and live_agent is not None:
    print("ANTES",live_agent.memory.get_block("human").value); print(live_agent.chat(PREFERENCE_CHANGE).response); print("DEPOIS",live_agent.memory.get_block("human").value)

# %%
def demonstrate_replace_delete(manager: EditableMemoryManager) -> None:
    manager.update_block("scratchpad","Preferência temporária: respostas longas. Remover depois.")
    memory_replace(manager,"scratchpad","respostas longas","respostas objetivas")
    memory_delete(manager,"scratchpad"," Remover depois.")
    print(manager.get_block("scratchpad").value)

# %%
TRIVIAL_MESSAGES=["Hoje está calor.","Obrigado.","Beleza.","Vou tomar café agora."]
# Memória boa é seletiva, não infinita. O teste real é opcional e não roda em import.

# %%
def should_remember(message: str) -> bool:
    durable_markers=("meu nome","prefiro","sou ","estou estudando","meu objetivo","sempre","não posso")
    normalized=message.lower().strip()
    return len(normalized)>12 and any(marker in normalized for marker in durable_markers)

# %%
class MemoryDecision(BaseModel):
    should_write: bool
    block: str | None = None
    operation: Literal["append","replace","delete","none"] = "none"
    reason: str
    new_content: str | None = None

# %%
def debug_memory_turn(agent: ContextMemoryAgent, message: str) -> TurnTrace:
    trace=agent.chat(message)
    print(json.dumps({"1_mensagem":message,"2_memoria_antes":trace.memory_before,"3_decisao":"inferida pelos tool calls","4_ferramentas":trace.tool_calls,"5_alteracoes":len(trace.tool_calls),"6_memoria_depois":trace.memory_after,"7_contexto":trace.context,"8_resposta":trace.response},ensure_ascii=False,indent=2))
    return trace

# %%
def context_size(system_prompt: str, manager: EditableMemoryManager, recent_messages: list[dict[str,str]]) -> pd.DataFrame:
    parts={"System Prompt":system_prompt,"Core Memory":manager.render_context(),"Recent Messages":json.dumps(recent_messages,ensure_ascii=False)}
    rows=[{"parte":k,"caracteres":len(v),"tokens_estimados":max(1,round(len(v)/4))} for k,v in parts.items()]
    rows.append({"parte":"Total","caracteres":sum(r["caracteres"] for r in rows),"tokens_estimados":sum(r["tokens_estimados"] for r in rows)})
    frame=pd.DataFrame(rows); print(frame.to_string(index=False)); return frame

# %%
def demonstrate_limit() -> str:
    manager=EditableMemoryManager([MemoryBlock(label="demo",description="Teste de limite",value="abc",limit=20)])
    manager.append_to_block("demo","123456")
    try: manager.append_to_block("demo","x"*30)
    except ValueError as e: return str(e)
    raise AssertionError("O limite deveria ter sido aplicado.")

# %%
def compact_memory_block(manager: EditableMemoryManager,label: str,target_chars: int,client: Groq | Any | None=None,model: str=GROQ_MODEL) -> MemoryBlock:
    block=manager.get_block(label)
    if target_chars<=0 or target_chars>block.limit: raise ValueError("target_chars deve ser positivo e não exceder o limite do bloco.")
    active_client=client or (Groq(api_key=GROQ_API_KEY) if GROQ_READY else None)
    if active_client is None: raise RuntimeError("GROQ_API_KEY ausente: compactação por LLM não executada.")
    response=active_client.chat.completions.create(model=model,messages=[{"role":"system","content":"Resuma a memória preservando fatos, preferências, restrições e decisões. Não invente nada."},{"role":"user","content":f"Limite máximo: {target_chars} caracteres.\n\n{block.value}"}])
    compacted=(response.choices[0].message.content or "").strip()
    if not compacted: raise ValueError("Resposta vazia durante compactação.")
    if len(compacted)>target_chars: raise ValueError(f"Compactação ainda excede o alvo: {len(compacted)}/{target_chars}.")
    return manager.update_block(label,compacted)

# %% [markdown]
# ## Core × Archival
# **Core memory** fica sempre disponível: nome, preferências e objetivos atuais.
# **Archival memory** fica fora do contexto e é recuperada sob demanda: histórico,
# documentos e fatos antigos.

# %%
class ArchivalMemory:
    def __init__(self,entries: list[str]|None=None): self.entries=entries or []
    def archive_memory(self,text: str) -> int:
        if not text.strip(): raise ValueError("Texto arquivado não pode ser vazio.")
        self.entries.append(text.strip()); return len(self.entries)-1
    @staticmethod
    def _tokens(text: str) -> set[str]: return set(re.findall(r"[\wÀ-ÿ]+",text.lower()))
    def search_archival_memory(self,query: str,k: int=3) -> list[dict[str,Any]]:
        q=self._tokens(query); results=[]
        for i,text in enumerate(self.entries):
            t=self._tokens(text); denom=np.sqrt(len(q)*len(t)); score=(len(q&t)/denom) if denom else 0.0
            results.append({"id":i,"text":text,"score":round(float(score),4)})
        return sorted(results,key=lambda x:x["score"],reverse=True)[:k]

archival_memory=ArchivalMemory()
def archive_memory(text: str) -> int: return archival_memory.archive_memory(text)
def search_archival_memory(query: str,k: int=3) -> list[dict[str,Any]]: return archival_memory.search_archival_memory(query,k)

# %% [markdown]
# ## Paging de memória
# Quando uma informação não precisa estar sempre ativa, ela pode sair da context
# window e ir para archival memory. Uma pergunta posterior dispara busca, recuperação
# e reinserção. É analogia com paging, não equivalência técnica literal.

# %%
ARCHIVED_STUDY="Nos estudos anteriores, A2A foi analisado como protocolo de descoberta e comunicação entre agentes, separado da lógica interna de cada agente."
archive_memory(ARCHIVED_STUDY)
if __name__ == "__main__": print(search_archival_memory("O que estudamos anteriormente sobre A2A?",k=1))

# %%
MEMORY_FILE=PROJECT_DIR/"agent_memory.json"
def save_memory(manager: EditableMemoryManager,path: Path=MEMORY_FILE,archive: ArchivalMemory|None=None) -> Path:
    payload={"blocks":manager.to_dict(),"archival":(archive or archival_memory).entries}
    try: path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    except OSError as e: raise OSError(f"Erro de persistência ao salvar {path}: {e}") from e
    return path
def load_memory(path: Path=MEMORY_FILE) -> tuple[EditableMemoryManager,ArchivalMemory]:
    try: data=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as e: raise ValueError(f"Erro de persistência/JSON em {path}: {e}") from e
    manager=EditableMemoryManager([MemoryBlock(label=label,**content) for label,content in data["blocks"].items()])
    return manager,ArchivalMemory(data.get("archival",[]))

# %%
def create_letta_agent(manager: EditableMemoryManager):
    if not LETTA_API_KEY: print("LETTA_API_KEY ausente; integração opcional ignorada."); return None,None
    try:
        from letta_client import Letta
        client=Letta(api_key=LETTA_API_KEY)
        state=client.agents.create(name="context-memory-academic",model=LETTA_MODEL,memory_blocks=[{"label":b.label,"description":b.description,"value":b.value,"limit":b.limit,"read_only":b.read_only} for b in manager.list_blocks() if b.label in {"persona","human","project"}])
        return client,state
    except Exception as e: raise RuntimeError(f"Erro do Letta ({type(e).__name__}): {e}") from e

# %%
def letta_read_block(client: Any,agent_id: str,label: str) -> str:
    return client.agents.blocks.retrieve(agent_id=agent_id,block_label=label).value
def letta_update_block(client: Any,agent_id: str,label: str,value: str) -> Any:
    return client.agents.blocks.update(agent_id=agent_id,block_label=label,value=value)

# %%
def letta_self_edit_turn(client: Any,agent_id: str,message: str) -> Any:
    if not message.strip(): raise ValueError("Mensagem vazia para Letta.")
    return client.agents.messages.create(agent_id=agent_id,messages=[{"role":"user","content":message}])

# %%
def render_agent_state(agent: ContextMemoryAgent) -> str:
    messages="\n".join(f"{m['role']}: {m['content']}" for m in agent.recent_messages) or "(vazio)"
    tools="\n".join(t["name"] for t in MEMORY_TOOLS)
    return f"AGENT STATE\n\nSYSTEM\n{SYSTEM_PROMPT}\n\nCORE MEMORY\n{agent.memory.render_context()}\n\nMESSAGES\n{messages}\n\nTOOLS\n{tools}\nmemory_search"

# %% [markdown]
# ## Context Management
# Decidir o que entra, permanece, sai, é compactado, arquivado e recuperado.
# **Context management não significa simplesmente aumentar a context window.**

# %% [markdown]
# ## LLM como sistema operacional
# A comparação `CPU/memória/storage/processos/paging` com
# `LLM/context window/external memory/tool calls/tasks/retrieval` serve para pensar
# em estado e recursos limitados. Uma LLM não é literalmente um sistema operacional.

# %% [markdown]
# ## Editable Memory × RAG
# Editable Memory modifica estado pessoal e preferências. RAG recupera conhecimento
# de documentos externos. Podem trabalhar juntos e têm responsabilidades diferentes.

# %% [markdown]
# ## Letta/MemGPT × LangGraph Store
# Letta/MemGPT enfatiza agentes persistentes e gestão explícita de contexto. LangGraph
# Store pode fornecer persistência a workflows LangGraph. Um não substitui integralmente o outro.

# %% [markdown]
# ## Memória × histórico de chat
# Histórico registra tudo que ocorreu. Memória seleciona informação estruturada útil.
# “Tomei café às 8h” não precisa persistir; uma restrição durável pode ser relevante
# conforme o domínio. Este exemplo não implementa decisões médicas.

# %%
def run_local_evaluation() -> pd.DataFrame:
    results=[]
    def add(test,expected,obtained,ok): results.append({"teste":test,"esperado":expected,"obtido":obtained,"status":"PASS" if ok else "FAIL"})
    m=load_seed(); before=m.get_block("human").value; m.append_to_block("human","Prefere exemplos práticos."); add("aprende preferência","bloco muda",m.get_block("human").value,m.get_block("human").value!=before)
    m.replace_in_block("human","Prefere exemplos práticos.","Prefere primeiro o exemplo."); add("substitui preferência","antiga ausente",m.get_block("human").value,"exemplos práticos" not in m.get_block("human").value)
    add("não salvar trivialidade","False",str(should_remember("Obrigado.")),not should_remember("Obrigado."))
    limit_msg=demonstrate_limit(); add("limite do bloco","erro claro",limit_msg,"excede" in limit_msg)
    a=ArchivalMemory(); a.archive_memory(ARCHIVED_STUDY); found=a.search_archival_memory("A2A protocolo agentes",1); add("recuperação externa","A2A",found[0]["text"],"A2A" in found[0]["text"])
    temp=PROJECT_DIR/"agent_memory.test.json"; save_memory(m,temp,a); restored,restored_archive=load_memory(temp); temp.unlink(); add("persistência/reload","mesmo human",restored.get_block("human").value,restored.get_block("human").value==m.get_block("human").value and bool(restored_archive.entries))
    for name in ["uso da preferência pela LLM","compactação por LLM","self-editing Letta"]: add(name,"depende de API","não executado sem chave",True)
    frame=pd.DataFrame(results); print(frame.to_string(index=False)); return frame

# %% [markdown]
# ## O que aprendi neste experimento
# A LLM não possui memória de aplicação automaticamente. O agente mantém estado fora
# do modelo, seleciona o que entra no contexto, edita fatos desatualizados, arquiva
# conteúdo menos ativo e o recupera quando necessário. Tool calling permite que o
# modelo solicite mudanças, mas a aplicação valida e executa cada operação.
#
# **A LLM raciocina sobre o contexto atual; o sistema de memória decide qual contexto
# estará disponível. Memória editável muda estado persistente, não os pesos da LLM.**

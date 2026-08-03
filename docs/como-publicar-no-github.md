# Como publicar no GitHub

## Opção pelo navegador

1. Entre em sua conta do GitHub.
2. Crie um repositório chamado `llms-agentes-ia`.
3. Não marque a criação automática de README, pois o projeto já possui um.
4. Extraia o arquivo ZIP deste projeto.
5. Na página vazia do repositório, escolha **uploading an existing file**.
6. Arraste os arquivos e pastas extraídos.
7. Escreva a mensagem `docs: adiciona estudos e experimentos do trilha de estudos`.
8. Confirme o commit.

## Opção pelo terminal

Abra o terminal dentro da pasta extraída:

```bash
git init
git add .
git commit -m "docs: adiciona estudos e experimentos do trilha de estudos"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/llms-agentes-ia.git
git push -u origin main
```

Substitua `SEU-USUARIO` pelo seu nome de usuário no GitHub.

## Sugestão de visibilidade

Comece como **privado** enquanto revisa informações pessoais, chaves, caminhos locais e anotações. Depois, torne público somente se estiver confortável.

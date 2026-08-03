"""Mensagem didática inspirada em protocolos de comunicação entre agentes."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import uuid

@dataclass
class ACPMessage:
    message_id: str
    task_id: str
    sender: str
    receiver: str
    message_type: str
    status: str
    payload: dict
    timestamp: str

    @classmethod
    def create(cls, *, task_id: str, sender: str, receiver: str,
               message_type: str, status: str, payload: dict) -> "ACPMessage":
        return cls(
            message_id=str(uuid.uuid4()),
            task_id=task_id,
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            status=status,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def run() -> None:
    task_id = "investigar-ioc-001"
    request = ACPMessage.create(
        task_id=task_id,
        sender="coordenador",
        receiver="agente_enriquecimento",
        message_type="task_request",
        status="requested",
        payload={"ioc": "203.0.113.42", "tipo": "ipv4"},
    )
    response = ACPMessage.create(
        task_id=task_id,
        sender="agente_enriquecimento",
        receiver="coordenador",
        message_type="task_result",
        status="completed",
        payload={
            "observacao": "Endereço reservado para documentação; exemplo fictício.",
            "confidence": 1.0,
        },
    )

    print("\n[ACP conceitual]")
    print(json.dumps(asdict(request), ensure_ascii=False, indent=2))
    print(json.dumps(asdict(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

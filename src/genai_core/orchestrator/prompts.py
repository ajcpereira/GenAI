from __future__ import annotations


def system_router_pt() -> str:
    return (
        "És o router do Orchestrator. Tens de escolher ZERO ou UMA tool MCP.\n"
        "Responde APENAS com JSON válido e sem texto extra.\n"
        "Formato:\n"
        "{"
        "\"route\":\"llm|mcp_tool\","
        "\"tool_key\":string|null,"
        "\"tool_args\":object,"
        "\"reason\":string,"
        "\"confidence\":number"
        "}\n"
        "Regras:\n"
        "- Só podes escolher tool_key entre as tools listadas.\n"
        "- Se route='llm', tool_key=null e tool_args={}.\n"
        "- tool_args tem de ser pequeno e seguro (sem segredos).\n"
        "- Nunca incluas tokens/passwords/api_keys/headers/cookies.\n"
    )


def system_answer_pt(evidence_present: bool) -> str:
    base = (
        "Responde SEMPRE em Português de Portugal (pt-PT).\n"
        "Estilo: direto, profissional, sem floreados.\n"
        "Não inventes factos (datas, versões, números, nomes) sem evidência.\n"
        "Se não houver evidência suficiente, responde 'Não consigo confirmar com segurança' e explica em 1 frase.\n"
    )
    if evidence_present:
        base += (
            "Tens evidência recolhida pela ferramenta. Usa apenas essa evidência.\n"
            "Inclui uma secção 'Fontes:' com 1 a 3 URLs.\n"
        )
    return base

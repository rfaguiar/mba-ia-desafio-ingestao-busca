from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SearchPromptTemplate:
    """Template for search prompts with context"""

    # Base template for search prompts
    BASE_TEMPLATE = """
        CONTEXTO:
        {context}

        REGRAS:
        - Responda somente com base no CONTEXTO fornecido.
        - Se a informação não estiver explicitamente no CONTEXTO, responda:
        "Não tenho informações necessárias para responder sua pergunta."
        - Nunca invente ou use conhecimento externo.
        - Nunca produza opiniões ou interpretações além do que está escrito.
        - Mantenha as respostas concisas e diretas.
        - Use linguagem clara e objetiva.

        EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
        Pergunta: "Qual é a capital da França?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        Pergunta: "Quantos clientes temos em 2024?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        Pergunta: "Você acha isso bom ou ruim?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        PERGUNTA DO USUÁRIO:
        {question}

        RESPONDA A "PERGUNTA DO USUÁRIO":
    """

    # Template for specific document types
    DOCUMENT_SPECIFIC_TEMPLATE = """
        CONTEXTO (Documento: {document_name}):
        {context}

        REGRAS:
        - Responda somente com base no CONTEXTO fornecido.
        - Se a informação não estiver explicitamente no CONTEXTO, responda:
        "Não tenho informações necessárias para responder sua pergunta."
        - Nunca invente ou use conhecimento externo.
        - Nunca produza opiniões ou interpretações além do que está escrito.
        - Mantenha as respostas concisas e diretas.
        - Use linguagem clara e objetiva.

        PERGUNTA DO USUÁRIO:
        {question}

        RESPONDA A "PERGUNTA DO USUÁRIO":
    """

    # Template for technical/scientific documents
    TECHNICAL_TEMPLATE = """
        CONTEXTO TÉCNICO:
        {context}

        REGRAS:
        - Responda somente com base no CONTEXTO fornecido.
        - Se a informação não estiver explicitamente no CONTEXTO, responda:
        "Não tenho informações necessárias para responder sua pergunta."
        - Use terminologia técnica apropriada quando presente no contexto.
        - Cite números, datas e fatos específicos quando disponíveis.
        - Mantenha precisão técnica nas respostas.
        - Nunca invente ou use conhecimento externo.

        PERGUNTA DO USUÁRIO:
        {question}

        RESPONDA A "PERGUNTA DO USUÁRIO":
    """

    # Template for business/financial documents
    BUSINESS_TEMPLATE = """
        CONTEXTO EMPRESARIAL:
        {context}

        REGRAS:
        - Responda somente com base no CONTEXTO fornecido.
        - Se a informação não estiver explicitamente no CONTEXTO, responda:
        "Não tenho informações necessárias para responder sua pergunta."
        - Use números e dados específicos quando disponíveis.
        - Cite datas e períodos quando relevantes.
        - Mantenha precisão nos valores financeiros e métricas.
        - Nunca invente ou use conhecimento externo.

        PERGUNTA DO USUÁRIO:
        {question}

        RESPONDA A "PERGUNTA DO USUÁRIO":
    """

    @classmethod
    def format_prompt(
        cls,
        question: str,
        context: List[Dict[str, Any]] = None,
        template_type: str = "base",
        document_name: Optional[str] = None,
    ) -> str:
        """Format prompt with context and question"""
        if not context:
            context_text = "Nenhum contexto fornecido."
        else:
            context_text = cls._format_context(context)

        # Select template based on type
        if template_type == "document_specific" and document_name:
            template = cls.DOCUMENT_SPECIFIC_TEMPLATE
            return template.format(
                context=context_text, question=question, document_name=document_name
            )
        elif template_type == "technical":
            template = cls.TECHNICAL_TEMPLATE
        elif template_type == "business":
            template = cls.BUSINESS_TEMPLATE
        else:
            template = cls.BASE_TEMPLATE

        return template.format(context=context_text, question=question)

    @classmethod
    def _format_context(cls, context: List[Dict[str, Any]]) -> str:
        """Format context data for prompt inclusion"""
        if not context:
            return "Nenhum contexto fornecido."

        formatted_parts = []
        for i, item in enumerate(context, 1):
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            # Add metadata information if available
            metadata_info = []
            if "page_number" in metadata:
                metadata_info.append(f"Página {metadata['page_number']}")
            if "chunk_index" in metadata:
                metadata_info.append(f"Chunk {metadata['chunk_index']}")

            metadata_str = f" ({', '.join(metadata_info)})" if metadata_info else ""

            formatted_parts.append(f"{i}. {content}{metadata_str}")

        return "\n\n".join(formatted_parts)

    @classmethod
    def create_summary_prompt(cls, context: List[Dict[str, Any]], question: str) -> str:
        """Create a prompt for summarizing context"""
        context_text = cls._format_context(context)

        summary_template = """
            CONTEXTO:
            {context}

            TAREFA:
            Crie um resumo conciso do contexto acima que responda à seguinte pergunta:

            PERGUNTA: {question}

            INSTRUÇÕES:
            - Baseie-se apenas no contexto fornecido
            - Seja conciso e direto
            - Organize as informações de forma lógica
            - Destaque os pontos mais relevantes para a pergunta

            RESUMO:
        """

        return summary_template.format(context=context_text, question=question)

    @classmethod
    def create_analysis_prompt(
        cls, context: List[Dict[str, Any]], question: str
    ) -> str:
        """Create a prompt for analyzing context"""
        context_text = cls._format_context(context)

        analysis_template = """
            CONTEXTO:
            {context}

            ANÁLISE SOLICITADA:
            {question}

            INSTRUÇÕES PARA ANÁLISE:
            - Analise o contexto fornecido
            - Identifique padrões, tendências ou insights relevantes
            - Forneça uma análise estruturada e objetiva
            - Baseie-se apenas nas informações disponíveis no contexto
            - Se não houver informações suficientes, indique claramente

            ANÁLISE:
        """

        return analysis_template.format(context=context_text, question=question)

    @classmethod
    def validate_prompt_length(cls, prompt: str, max_length: int = 32000) -> bool:
        """Validate if prompt is within length limits"""
        return len(prompt) <= max_length

    @classmethod
    def truncate_context_if_needed(
        cls, context: List[Dict[str, Any]], question: str, max_length: int = 32000
    ) -> List[Dict[str, Any]]:
        """Truncate context if prompt would be too long"""
        # Start with full context
        test_context = context.copy()

        while test_context:
            test_prompt = cls.format_prompt(question, test_context)

            if cls.validate_prompt_length(test_prompt, max_length):
                break

            # Remove last context item
            test_context.pop()

        if len(test_context) < len(context):
            logger.warning(
                f"Context truncated from {len(context)} to {len(test_context)} items due to length limits"
            )

        return test_context

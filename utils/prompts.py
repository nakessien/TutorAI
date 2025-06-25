from typing import Dict, List, Optional, Any
from enum import Enum


class ResponseStyle(Enum):
    """Response style enumeration"""
    BALANCED = "balanced"
    DETAILED_POLICY = "detailed_policy"
    PRACTICAL_GUIDE = "practical_guide"


class PromptTemplates:
    """Prompt template manager"""

    SYSTEM_BASE = """You are an intelligent academic advisor assistant designed to help university tutors efficiently respond to student policy inquiries.

Your core responsibilities:
1. Analyze policy documents and provide accurate, helpful responses
2. Adapt your communication style based on the specified approach
3. Maintain professionalism while being approachable and clear
4. Cite relevant policy sections when appropriate
5. Provide actionable guidance that students can follow

IMPORTANT: Keep your response concise and focused. Limit your answer to approximately 200 words or less.

Guidelines:
- Always base your responses on the provided policy context
- Be factually accurate and avoid speculation
- Structure your responses logically and clearly
- Use appropriate tone for the requested style
- Include specific next steps when relevant
- BE CONCISE - Quality over quantity"""

    RAG_CONTEXT_TEMPLATE = """Based on the following relevant policy documents, please answer the student's question:

RELEVANT POLICY CONTEXT:
{retrieved_context}

STUDENT QUESTION:
{question}

Please provide a response that references the specific policies above and addresses the student's concern directly. 
REMEMBER: Keep your response under 200 words while maintaining clarity and usefulness."""

    STYLE_BALANCED = """RESPONSE STYLE: BALANCED

Provide a response that:
- Combines official policy information with practical guidance
- Uses professional but approachable language
- Balances thoroughness with clarity and readability
- Includes both policy requirements AND practical next steps
- Cites relevant policies while explaining their practical implications
- Maintains a helpful, supportive tone

Structure your response with:
1. Direct answer to the question (1-2 sentences)
2. Key policy points (2-3 bullet points)
3. Practical next steps (2-3 action items)

IMPORTANT: Keep the entire response under 200 words.

Tone: Professional yet warm, comprehensive but accessible."""

    STYLE_DETAILED_POLICY = """RESPONSE STYLE: DETAILED POLICY

Provide a response that:
- Focuses heavily on official policy language and exact requirements
- Uses formal academic/administrative language
- Emphasizes compliance, procedures, and official protocols
- Includes specific policy citations, section numbers, and references
- Provides comprehensive coverage of relevant policy aspects
- Prioritizes accuracy and completeness

Structure your response with:
1. Specific policy citations and references
2. Key requirements and conditions (bullet points)
3. Compliance considerations and deadlines

IMPORTANT: Even with detailed policy focus, keep response under 200 words by being selective about the most critical information.

Tone: Formal, authoritative, policy-focused, precise."""

    STYLE_PRACTICAL_GUIDE = """RESPONSE STYLE: PRACTICAL GUIDE

Provide a response that:
- Focuses on actionable steps and clear instructions
- Uses simple, conversational language that students easily understand
- Emphasizes "how to" rather than "what is"
- Provides step-by-step guidance with specific actions
- Minimizes policy jargon and maximizes practical clarity
- Includes specific deadlines, contact information, and next actions

Structure your response with:
1. Clear, direct answer (1 sentence)
2. Step-by-step action plan (3-5 numbered steps)
3. Key contact or deadline (1 line)

IMPORTANT: Keep response under 200 words - focus on the most essential actions only.

Tone: Helpful, clear, action-oriented, student-friendly."""

    def get_complete_prompt(self,
                            style: ResponseStyle,
                            question: str,
                            context: str,
                            user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Generate complete prompt for given style and context"""
        prompt_parts = []

        prompt_parts.append(self.SYSTEM_BASE)

        style_prompts = {
            ResponseStyle.BALANCED: self.STYLE_BALANCED,
            ResponseStyle.DETAILED_POLICY: self.STYLE_DETAILED_POLICY,
            ResponseStyle.PRACTICAL_GUIDE: self.STYLE_PRACTICAL_GUIDE
        }

        if style in style_prompts:
            prompt_parts.append(style_prompts[style])

        if user_preferences:
            preference_adjustment = self._generate_preference_adjustment(user_preferences)
            if preference_adjustment:
                prompt_parts.append(preference_adjustment)

        rag_section = self.RAG_CONTEXT_TEMPLATE.format(
            retrieved_context=context,
            question=question
        )
        prompt_parts.append(rag_section)

        # 添加最后的长度提醒
        prompt_parts.append("\nREMINDER: Provide a helpful, complete answer in 200 words or less.\n\nRESPONSE:")

        return "\n\n".join(prompt_parts)

    def _generate_preference_adjustment(self, preferences: Dict[str, Any]) -> str:
        """Generate preference-based adjustments"""
        adjustments = []

        if preferences.get("prefers_shorter_responses"):
            adjustments.append("Keep your response especially concise (aim for 100-150 words)")
        elif preferences.get("prefers_longer_responses"):
            adjustments.append("Provide a comprehensive response (but still within 200 words)")

        if preferences.get("prefers_more_citations"):
            adjustments.append("Include specific policy citations and reference numbers")
        elif preferences.get("prefers_fewer_citations"):
            adjustments.append("Focus on practical information rather than detailed policy citations")

        if preferences.get("prefers_examples"):
            adjustments.append("Include a brief example if space permits")

        if preferences.get("prefers_step_by_step"):
            adjustments.append("Break down complex processes into numbered, sequential steps")

        formality = preferences.get("formality_level")
        if formality == "more_formal":
            adjustments.append("Use formal, professional language throughout")
        elif formality == "less_formal":
            adjustments.append("Use friendly, conversational language while maintaining professionalism")

        if preferences.get("prefers_contact_info"):
            adjustments.append("Include relevant contact information when applicable")

        if adjustments:
            return "ADDITIONAL PREFERENCES:\n" + "\n".join(f"- {adj}" for adj in adjustments)

        return ""

    STYLE_TRANSITION_PROMPTS = {
        f"{ResponseStyle.BALANCED.value}_to_{ResponseStyle.DETAILED_POLICY.value}":
            "Now provide a more policy-focused response (still under 200 words):",

        f"{ResponseStyle.BALANCED.value}_to_{ResponseStyle.PRACTICAL_GUIDE.value}":
            "Here's the same information in a more step-by-step format (under 200 words):",

        f"{ResponseStyle.DETAILED_POLICY.value}_to_{ResponseStyle.BALANCED.value}":
            "Let me provide a more balanced view (under 200 words):",

        f"{ResponseStyle.DETAILED_POLICY.value}_to_{ResponseStyle.PRACTICAL_GUIDE.value}":
            "Here's the practical guide version (under 200 words):",

        f"{ResponseStyle.PRACTICAL_GUIDE.value}_to_{ResponseStyle.BALANCED.value}":
            "Let me provide a more comprehensive view (under 200 words):",

        f"{ResponseStyle.PRACTICAL_GUIDE.value}_to_{ResponseStyle.DETAILED_POLICY.value}":
            "Here's the policy-focused version (under 200 words):"
    }

    def get_style_transition_prompt(self,
                                    from_style: ResponseStyle,
                                    to_style: ResponseStyle,
                                    original_question: str,
                                    context: str) -> str:
        """Generate style transition prompt"""
        transition_key = f"{from_style.value}_to_{to_style.value}"
        transition_phrase = self.STYLE_TRANSITION_PROMPTS.get(
            transition_key,
            f"Here's the same information presented in a {to_style.value.replace('_', ' ')} style (under 200 words):"
        )

        switch_prompt = f"{transition_phrase}\n\n{self.get_complete_prompt(to_style, original_question, context)}"

        return switch_prompt


class SpecialScenarioPrompts:
    """Special scenario prompt templates"""

    INSUFFICIENT_CONTEXT = """I don't have sufficient information in the available policy documents to fully answer your specific question about: "{question}"

However, I can provide some general guidance:

**Immediate Next Steps:**
1. Contact the Academic Affairs Office directly for specific requirements
2. Check the student handbook for updates
3. Schedule an appointment with your academic advisor

**General Information:**
{general_guidance}

**Contact Information:**
- Academic Affairs Office: [Contact details]
- Student Services: [Contact details]

Would you like help with a different question?"""

    ERROR_RECOVERY = """I apologize, but I encountered an issue while processing your request. 

**Alternative approaches:**
1. **Rephrase your question**: Try different wording
2. **Break down complex questions**: Ask about one aspect at a time
3. **Contact directly**: Reach out to the relevant office

**Direct assistance:**
- Academic Affairs Office: [contact]
- Student Services: [contact]

What would be most helpful for you right now?"""


def get_next_style_in_sequence(current_styles: List[str]) -> str:
    """Get next style in fixed sequence: balanced -> detailed_policy -> practical_guide"""
    sequence = ["balanced", "detailed_policy", "practical_guide"]
    next_index = len(current_styles) % len(sequence)
    return sequence[next_index]
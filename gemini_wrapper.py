"""
Gemini API Wrapper for RAG Systems
"""

import google.genai as genai
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiWrapper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        verbose: bool = True
    ):
        # Use provided key OR environment key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        # Check if missing
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Either pass api_key parameter or set GEMINI_API_KEY environment variable.\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )

        # Create Gemini client
        self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        self.history = []
        self.persona = None   # start empty

        if self.verbose:
            print(f"✅ Gemini initialized: {model_name} (temp={temperature})")

    def set_persona(self, persona_description: str) -> None:
        self.persona = persona_description

        if self.verbose:
            preview = persona_description[:80] + "..." if len(persona_description) > 80 else persona_description
            print(f"✅ Persona set: {preview}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2048
    ) -> str:

        full_prompt = (
            f"SYSTEM: {self.persona}\n\nUSER: {prompt}"
            if self.persona else prompt
        )

        temp = temperature if temperature is not None else self.temperature

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
            )

            text = ""
            if hasattr(resp, "text") and isinstance(resp.text, str):
                text = resp.text
            elif hasattr(resp, "candidates") and resp.candidates:
                for cand in resp.candidates:
                    if getattr(cand, "content", None):
                        parts = getattr(cand.content, "parts", [])
                        for p in parts:
                            if getattr(p, "text", None):
                                text += p.text
                text = text.strip()

            self.history.append({
                'prompt': prompt,
                'response': text,
                'temperature': temp,
                'model': self.model_name
            })

            return text or ""

        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg

    def chat(self, message: str) -> str:
        if not hasattr(self, '_chat_transcript'):
            self._chat_transcript = []

        self._chat_transcript.append({"role": "user", "text": message})

        convo = []
        if self.persona:
            convo.append(f"SYSTEM: {self.persona}")

        for turn in self._chat_transcript[-10:]:
            prefix = "USER" if turn["role"] == "user" else "ASSISTANT"
            convo.append(f"{prefix}: {turn['text']}")

        convo.append("ASSISTANT:")
        prompt = "\n\n".join(convo)

        reply = self.generate(prompt)

        self._chat_transcript.append({"role": "assistant", "text": reply})
        return reply

    def clear_history(self) -> None:
        self.history = []
        if hasattr(self, '_chat_transcript'):
            self._chat_transcript = []
        if self.verbose:
            print("✅ History cleared")

    def get_history(self) -> List[Dict]:
        return self.history

    def get_stats(self) -> Dict:
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'total_interactions': len(self.history),
            'has_persona': self.persona is not None
        }


# Demo
def demo():
    print("\n" + "="*70)
    print("GEMINI WRAPPER DEMO")
    print("="*70 + "\n")

    try:
        llm = GeminiWrapper()

        print("1. Basic Generation")
        print("-" * 70)
        response = llm.generate("What is Python in one sentence?")
        print("A:", response, "\n")

        print("2. With Persona")
        print("-" * 70)
        llm.set_persona(
            "You are a helpful teacher who explains concepts using simple analogies."
        )
        response = llm.generate("What is machine learning?")
        print("A:", response, "\n")

        print("3. Chat Mode")
        print("-" * 70)
        llm.chat("My favorite color is blue")
        reply = llm.chat("What's my favorite color?")
        print("AI:", reply)

    except ValueError as e:
        print("\n❌ Error:", e)
        print("\nCreate .env file:")
        print("GEMINI_API_KEY=your_key_here\n")


if __name__ == "__main__":
    demo()
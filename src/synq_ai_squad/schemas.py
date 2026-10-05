"""The structured outputs the agents must return. Pydantic checks their shape for us."""

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """What the Manager must return."""

    topic: str = Field(description="A clear, specific topic for the piece")
    content_format: str = Field(description='e.g. "LinkedIn post", "short blog article", "email newsletter"')
    audience: str = Field(description="Who will read it, as specifically as possible")
    angle: str = Field(description="The main message or hook, in one sentence")
    search_queries: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3-5 short, different searches to run against the company documents",
    )


class CheckedClaim(BaseModel):
    claim: str = Field(description="One specific claim the draft makes about Synq Logic, quoted from the draft")
    evidence: str = Field(
        description="The line from the research notes that backs it up, quoted exactly, or '' if none"
    )
    supported: bool = Field(description="False if there's no evidence, or the claim ADDS any detail the evidence lacks")


class Review(BaseModel):
    """What the Critic must return. Field order matters: the model fills them top to bottom,
    so it checks the claims first and only then decides the score."""

    claims: list[CheckedClaim]
    issues: list[str] = Field(description="Other problems to fix (clarity, value, format). Empty if none.")
    score: int = Field(ge=1, le=10, description="Overall quality from 1 (bad) to 10 (ready to publish)")

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]

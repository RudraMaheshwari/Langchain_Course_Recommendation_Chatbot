class ConversationState:
    def __init__(self) -> None:
        self.grade: int | None = None
        self.interests: list[str] = []
        self.interest_turns: int = 0 

    def set_grade(self, grade: int) -> None:
        if not isinstance(grade, int):
            raise ValueError("Grade must be an integer.")
        if not 8 <= grade <= 12:
            raise ValueError("Grade must be between 9 and 12.")
        self.grade = grade

    def add_interest(self, interest: str) -> bool:
        if interest and interest.lower() not in [i.lower() for i in self.interests]:
            self.interests.append(interest)
            return True
        return False

    def remove_interest(self, interest: str) -> None:
        interest = interest.strip().lower()
        if interest in self._get_lower_interests():
            self.interests.remove(self._find_interest(interest))

    def has_interest(self, interest: str) -> bool:
        interest = interest.strip().lower()
        return interest in self._get_lower_interests()

    def get_grade(self) -> int | None:
        return self.grade

    def get_interests(self) -> str:
        return ", ".join(self.interests)

    def get_interests_list(self) -> list[str]:
        return self.interests

    def is_fully_set(self) -> bool:
        return self.grade is not None and bool(self.interests)

    def reset(self) -> None:
        self.grade = None
        self.interests = []

    def _get_lower_interests(self) -> list[str]:
        return [interest.lower() for interest in self.interests]

    def _find_interest(self, interest: str) -> str:
        lower_interests = self._get_lower_interests()
        if interest.lower() not in lower_interests:
            raise ValueError(f"Interest '{interest}' not found.")
        return self.interests[lower_interests.index(interest.lower())]

    def __str__(self) -> str:
        return f"ConversationState(grade={self.grade}, interests={self.interests})"

    def reset(self) -> None:
        self.grade = None
        self.interests = []
        self.interest_turns = 0

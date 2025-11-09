from transitions import Machine
from graphviz import Digraph


class EmotionFSM:
    """Finite-State Machine managing the bot's emotional states."""

    states = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful", "Curious"]

    transitions_def = [
        {"trigger": "to_happy", "source": "*", "dest": "Happy"},
        {"trigger": "to_sad", "source": "*", "dest": "Sad"},
        {"trigger": "to_angry", "source": "*", "dest": "Angry"},
        {"trigger": "to_neutral", "source": "*", "dest": "Neutral"},
        {"trigger": "to_surprised", "source": "*", "dest": "Surprised"},
        {"trigger": "to_fearful", "source": "*", "dest": "Fearful"},
        {"trigger": "to_curious", "source": "*", "dest": "Curious"},
    ]

    def __init__(self):
        self.machine = Machine(
            model=[self],
            states=self.states,
            transitions=self.transitions_def,
            initial="Neutral",
            auto_transitions=False,
        )
        self.mood_score = {s: 0.0 for s in self.states}

    # --------------------------------------------------------
    # visualization
    # --------------------------------------------------------
    def get_graphviz_source(self):
        """Render FSM highlighting the current state."""
        dot = Digraph(format="png")
        dot.attr(rankdir="LR")

        current = getattr(self, "state", "Neutral")

        for s in self.states:
            if s == current:
                dot.node(s, style="filled", color="lightblue")
            else:
                dot.node(s)

        for s in self.states:
            for t in self.states:
                if s != t:
                    dot.edge(s, t, arrowhead="vee")

        return dot.source

    # --------------------------------------------------------
    # logic
    # --------------------------------------------------------
    def update_from_nlp(self, emotion_scores: dict, sentiment_scores: dict):
        """Update state using emotion/sentiment outputs."""
        top_emo = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else None
        top_sent = max(sentiment_scores.items(), key=lambda x: x[1])[0] if sentiment_scores else None

        mapping = {
            "joy": "Happy", "happy": "Happy",
            "sadness": "Sad", "sad": "Sad",
            "anger": "Angry", "angry": "Angry",
            "surprise": "Surprised", "surprised": "Surprised",
            "fear": "Fearful", "fearful": "Fearful",
            "neutral": "Neutral",
            "disgust": "Angry", "trust": "Curious",
            "anticipation": "Curious", "curious": "Curious",
        }

        chosen_state = None
        if top_emo:
            chosen_state = mapping.get(top_emo.lower())
        if not chosen_state and top_sent:
            if top_sent.upper() == "POSITIVE":
                chosen_state = "Happy"
            elif top_sent.upper() == "NEGATIVE":
                chosen_state = "Sad"
            else:
                chosen_state = "Neutral"

        self._apply_transition(chosen_state or "Neutral")
        return self.state

    def _apply_transition(self, target_state: str):
        """Fire the appropriate trigger for a given target state."""
        mapping = {
            "Happy": self.to_happy,
            "Sad": self.to_sad,
            "Angry": self.to_angry,
            "Neutral": self.to_neutral,
            "Surprised": self.to_surprised,
            "Fearful": self.to_fearful,
            "Curious": self.to_curious,
        }
        trigger = mapping.get(target_state, self.to_neutral)
        trigger()

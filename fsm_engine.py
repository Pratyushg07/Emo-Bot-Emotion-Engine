from transitions import Machine
from graphviz import Digraph
import math

class EmotionFSM:
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
            model=self,
            states=EmotionFSM.states,
            transitions=EmotionFSM.transitions_def,
            initial="Neutral"
        )
        self.mood_score = {s: 0.0 for s in EmotionFSM.states}

    def get_graphviz_source(self):
        dot = Digraph(format='png')
        dot.attr(rankdir='LR')
        for s in EmotionFSM.states:
            if s == self.state:
                dot.node(s, style="filled", color="lightblue")
            else:
                dot.node(s)
        for s in EmotionFSM.states:
            for t in EmotionFSM.states:
                dot.edge(s, t, arrowhead='vee')
        return dot.source

    def update_from_nlp(self, emotion_scores: dict, sentiment_scores: dict):
        if not emotion_scores:
            top_emo = None
        else:
            top_emo = max(emotion_scores.items(), key=lambda x: x[1])[0]

        if sentiment_scores:
            top_sent = max(sentiment_scores.items(), key=lambda x: x[1])[0]
        else:
            top_sent = None

        mapping = {
            "joy": "Happy", "happy": "Happy", "happiness": "Happy",
            "sadness": "Sad", "sad": "Sad",
            "anger": "Angry", "angry": "Angry",
            "surprise": "Surprised", "surprised": "Surprised",
            "fear": "Fearful", "fearful": "Fearful",
            "neutral": "Neutral",
            "disgust": "Angry", "trust": "Curious", "anticipation": "Curious",
            "curiosity": "Curious", "curious": "Curious"
        }

        chosen_state = None

        if top_emo:
            t = top_emo.lower()
            chosen_state = mapping.get(t, None)

        if not chosen_state and top_sent:
            if top_sent.upper() == "POSITIVE":
                chosen_state = "Happy"
            elif top_sent.upper() == "NEGATIVE":
                chosen_state = "Sad"
            else:
                chosen_state = "Neutral"

        if chosen_state:
            self._apply_transition(chosen_state)
        else:
            self.to_neutral()

        return self.state

    def _apply_transition(self, target_state):
        trigger = {
            "Happy": self.to_happy,
            "Sad": self.to_sad,
            "Angry": self.to_angry,
            "Neutral": self.to_neutral,
            "Surprised": self.to_surprised,
            "Fearful": self.to_fearful,
            "Curious": self.to_curious
        }.get(target_state, self.to_neutral)
        trigger()

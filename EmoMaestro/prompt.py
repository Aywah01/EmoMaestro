# prompt.py (3D VAD version)

import random
from typing import Dict, Tuple

class Prompt:
    """
    3D VAD-based prompt generator for MusicGen.

    Input:
        vad = {
            "valence":   float in [-1, 1],
            "arousal":   float in [-1, 1],
            "dominance": float in [-1, 1],
        }

    Output:
        A natural-language prompt string.
    """

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        # 8 octant profiles: (sign_v, sign_a, sign_d) -> profile dict
        # sign: +1 = high/positive, -1 = low/negative
        self._profiles: Dict[Tuple[int, int, int], Dict] = {
            # +V +A +D : very positive, energetic, powerful → EDM / Pop Rock
            (+1, +1, +1): {
                "name": "Triumphant Euphoric Energy",
                "genres": ["EDM", "festival house", "pop rock", "future bass"],
                "mood_words": [
                    "euphoric", "triumphant", "uplifting", "celebratory"
                ],
                "tempo_words": [
                    "fast tempo", "driving beat", "high energy rhythm"
                ],
                "instrument_words": [
                    "bright synths", "punchy drums", "wide supersaw leads"
                ]
            },
            # +V +A -D : positive, energetic, but gentle/soft → K-pop, dance pop
            (+1, +1, -1): {
                "name": "Playful Excited Lightness",
                "genres": ["dance pop", "K-pop style", "nu-disco", "tropical house"],
                "mood_words": [
                    "playful", "excited", "cheerful", "bubbly"
                ],
                "tempo_words": [
                    "bouncy groove", "upbeat tempo", "rhythmic and catchy"
                ],
                "instrument_words": [
                    "sparkling synths", "tight drum patterns", "light plucks"
                ]
            },
            # +V -A +D : positive but calm and confident → chillhop / lo-fi / soft rock
            (+1, -1, +1): {
                "name": "Calm Confident Warmth",
                "genres": ["chillhop", "lo-fi beats", "soft rock", "chill R&B"],
                "mood_words": [
                    "warm", "peaceful", "confident", "reassuring"
                ],
                "tempo_words": [
                    "steady mid tempo", "laid-back groove", "smooth rhythmic flow"
                ],
                "instrument_words": [
                    "soft electric piano", "warm bass", "gentle percussion"
                ]
            },
            # +V -A -D : positive, calm, and delicate → ambient / acoustic
            (+1, -1, -1): {
                "name": "Gentle Relaxed Comfort",
                "genres": ["ambient", "acoustic", "cinematic soft score"],
                "mood_words": [
                    "gentle", "comforting", "relaxed", "soothing"
                ],
                "tempo_words": [
                    "slow tempo", "minimal rhythm", "flowing and airy"
                ],
                "instrument_words": [
                    "soft piano", "light acoustic guitar", "smooth pads"
                ]
            },
            # -V +A +D : negative valence, high arousal, powerful → aggressive metal / industrial
            (-1, +1, +1): {
                "name": "Aggressive Intense Power",
                "genres": ["heavy metal", "industrial", "hard rock", "dark techno"],
                "mood_words": [
                    "aggressive", "intense", "furious", "darkly energetic"
                ],
                "tempo_words": [
                    "very fast tempo", "relentless rhythm", "driving pulse"
                ],
                "instrument_words": [
                    "distorted guitars", "heavy drums", "dark synths"
                ]
            },
            # -V +A -D : negative, high arousal, low dominance → fear, panic, horror
            (-1, +1, -1): {
                "name": "Panicked Tense Horror",
                "genres": ["horror soundtrack", "dark ambient", "cinematic tension"],
                "mood_words": [
                    "tense", "frightened", "restless", "uneasy"
                ],
                "tempo_words": [
                    "irregular pulses", "sharp accents", "nervous rhythmic motion"
                ],
                "instrument_words": [
                    "dissonant strings", "stuttering synths", "deep drones"
                ]
            },
            # -V -A +D : negative, low arousal, high dominance → heavy but controlled sadness
            (-1, -1, +1): {
                "name": "Dark Stoic Weight",
                "genres": ["slow metal", "dark jazz", "moody soundtrack"],
                "mood_words": [
                    "brooding", "stoic", "heavy", "melancholic"
                ],
                "tempo_words": [
                    "slow deliberate tempo", "steady but heavy rhythm"
                ],
                "instrument_words": [
                    "low strings", "dark piano chords", "slow drums"
                ]
            },
            # -V -A -D : negative, low arousal, low dominance → fragile sadness / depression
            (-1, -1, -1): {
                "name": "Fragile Deep Sadness",
                "genres": ["piano ballad", "sad ambient", "minimal soundtrack"],
                "mood_words": [
                    "fragile", "deeply sad", "intimate", "reflective"
                ],
                "tempo_words": [
                    "very slow tempo", "gentle, sparse rhythm", "lingering phrases"
                ],
                "instrument_words": [
                    "solo piano", "subtle strings", "soft reverb tails"
                ]
            },
        }

    def _sign(self, x: float) -> int:
        """
        Convert a value in [-1, 1] into +1 or -1.
        Values close to 0 are nudged using eps.
        """
        if x >= 0:
            return +1
        else:
            return -1

    def _get_octant(self, v: float, a: float, d: float) -> Tuple[int, int, int]:
        sv = self._sign(v)
        sa = self._sign(a)
        sd = self._sign(d)
        return sv, sa, sd

    def _intensity_adverb(self, x: float) -> str:
        """
        Map |x| to an intensity adverb.
        """
        mag = abs(x)
        if mag < 0.25:
            return "slightly"
        elif mag < 0.55:
            return "moderately"
        elif mag < 0.8:
            return "strongly"
        else:
            return "extremely"

    def gen(self, vad, extra_tags: str | None = None) -> str:
        """
        Generate a natural language prompt given a VAD dict.
        """
        v = float(vad[0])
        a = float(vad[1])
        d = float(vad[2])

        octant = self._get_octant(v, a, d)
        profile = self._profiles.get(octant)

        if profile is None:
            raise Exception(f"No profile found")

        genre = random.choice(profile["genres"])
        mood_word = random.choice(profile["mood_words"])
        tempo_word = random.choice(profile["tempo_words"])
        instr_word = random.choice(profile["instrument_words"])

        v_int = self._intensity_adverb(v)
        a_int = self._intensity_adverb(a)
        d_int = self._intensity_adverb(d)

        dominance_phrase = "powerful and commanding" if d >= 0 else "fragile and vulnerable"

        base_prompt = (
            f"{genre} instrumental soundtrack, {v_int} {mood_word} mood, "
            f"{a_int} energetic feeling with {tempo_word}, "
            f"feels {dominance_phrase}, featuring {instr_word}, "
            "high quality production, no vocals."
        )

        if extra_tags:
            base_prompt += " " + extra_tags

        return base_prompt

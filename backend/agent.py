"""
AI Investigator Agent for deepfake detection analysis.

Provides natural-language explanations of detection results through
rule-based pattern analysis and optional Google Gemini API integration.
"""

import logging
import os
import statistics
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeepfakeInvestigator:
    """
    Generates human-readable analysis reports from deepfake detection results.

    Operates in two modes:
        1. **Rule-based** (default): Examines confidence scores, model agreement,
           temporal patterns, and spatial anomalies to produce detailed explanations.
        2. **Gemini-enhanced**: When GEMINI_API_KEY is set, augments rule-based
           analysis with Google Gemini LLM for richer, more nuanced explanations.
    """

    # Confidence thresholds for verdict categorization
    THRESHOLDS = {
        "very_high": 0.90,
        "high": 0.75,
        "moderate": 0.50,
        "low": 0.25,
        "very_low": 0.10,
    }

    MODEL_DESCRIPTIONS = {
        "efficientnet": (
            "EfficientNet (spatial analysis specialist) — excels at detecting "
            "pixel-level artifacts, blending boundaries, and texture inconsistencies"
        ),
        "vit": (
            "Vision Transformer (global context analyst) — captures long-range "
            "dependencies and structural coherence across the entire face"
        ),
        "cnn_lstm": (
            "CNN-LSTM (temporal dynamics tracker) — analyzes frame-to-frame "
            "consistency and detects temporal artifacts in video sequences"
        ),
    }

    def __init__(self):
        self._gemini_model = None
        self._gemini_available = False
        self._init_gemini()

    # ------------------------------------------------------------------ #
    #  Gemini initialization
    # ------------------------------------------------------------------ #

    def _init_gemini(self) -> None:
        """Attempt to initialize Google Gemini if API key is available."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.info("GEMINI_API_KEY not set — using rule-based analysis only.")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            self._gemini_available = True
            logger.info("Gemini integration enabled (model: gemini-2.0-flash).")
        except ImportError:
            logger.warning(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.warning("Failed to initialize Gemini: %s", e)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def analyze(self, detection_results: dict) -> str:
        """
        Generate a comprehensive analysis report from detection results.

        Args:
            detection_results: Dict containing verdict, confidence,
                per_model_scores, frame_analysis, and metadata.

        Returns:
            Detailed natural-language explanation string.
        """
        # Always build rule-based analysis first
        rule_analysis = self._rule_based_analysis(detection_results)

        if self._gemini_available:
            try:
                return self._gemini_enhanced_analysis(detection_results, rule_analysis)
            except Exception as e:
                logger.warning("Gemini analysis failed, falling back: %s", e)

        return rule_analysis

    def chat(self, message: str, context: Optional[dict] = None) -> str:
        """
        Conversational interface for follow-up questions about detection results.

        Args:
            message: User's question or message.
            context: Optional detection results context.

        Returns:
            Agent's response string.
        """
        if self._gemini_available and context:
            try:
                return self._gemini_chat(message, context)
            except Exception as e:
                logger.warning("Gemini chat failed, falling back: %s", e)

        return self._rule_based_chat(message, context)

    # ------------------------------------------------------------------ #
    #  Rule-based analysis engine
    # ------------------------------------------------------------------ #

    def _rule_based_analysis(self, results: dict) -> str:
        """Produce a detailed rule-based analysis from detection results."""
        sections: list[str] = []

        verdict = results.get("verdict", "UNKNOWN")
        confidence = results.get("confidence", 0.0)
        per_model = results.get("per_model_scores", {})
        frame_analysis = results.get("frame_analysis", [])
        metadata = results.get("metadata", {})

        # 1. Executive summary
        sections.append(self._build_summary(verdict, confidence, metadata))

        # 2. Per-model breakdown
        if per_model:
            sections.append(self._build_model_breakdown(per_model, confidence))

        # 3. Model agreement analysis
        if per_model:
            sections.append(self._build_agreement_analysis(per_model))

        # 4. Temporal analysis (if multiple frames)
        if len(frame_analysis) > 1:
            sections.append(self._build_temporal_analysis(frame_analysis))

        # 5. Spatial anomaly insights
        sections.append(self._build_spatial_insights(per_model, confidence, verdict))

        # 6. Conclusion and confidence assessment
        sections.append(self._build_conclusion(verdict, confidence, per_model))

        return "\n\n".join(sections)

    # --- Section builders ------------------------------------------------- #

    def _build_summary(self, verdict: str, confidence: float, metadata: dict) -> str:
        filename = metadata.get("filename", "uploaded media")
        input_type = metadata.get("input_type", "file")
        num_frames = metadata.get("num_frames", 1)

        confidence_pct = confidence * 100
        severity = self._confidence_label(confidence)

        lines = [
            "🔍 **DEEPFAKE ANALYSIS REPORT**",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"**File:** {filename} ({input_type})",
            f"**Frames analyzed:** {num_frames}",
            f"**Verdict:** {verdict}",
            f"**Manipulation confidence:** {confidence_pct:.1f}% ({severity})",
            "",
        ]

        if verdict == "FAKE":
            lines.append(
                f"⚠️  The analysis indicates this {input_type} has likely been "
                f"digitally manipulated with {severity} confidence."
            )
        elif verdict == "REAL":
            lines.append(
                f"✅  The analysis indicates this {input_type} appears to be "
                f"authentic with {severity} confidence."
            )
        else:
            lines.append(
                "⚡  The analysis is inconclusive — the confidence level falls "
                "within the uncertain range."
            )

        return "\n".join(lines)

    def _build_model_breakdown(self, per_model: dict, ensemble_conf: float) -> str:
        lines = [
            "📊 **MODEL-BY-MODEL BREAKDOWN**",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        for model_key, score in per_model.items():
            if score is None:
                description = self.MODEL_DESCRIPTIONS.get(
                    model_key, f"{model_key} model"
                )
                lines.append(f"• **{model_key.upper()}** — N/A")
                lines.append(f"  {description}")
                lines.append("  → Not applicable for this input (e.g. single image, no temporal data)")
                lines.append("")
                continue

            pct = score * 100
            description = self.MODEL_DESCRIPTIONS.get(
                model_key, f"{model_key} model"
            )
            bar = self._confidence_bar(score)

            delta = score - ensemble_conf
            delta_str = f"({'+' if delta >= 0 else ''}{delta * 100:.1f}% vs ensemble)"

            lines.append(f"• **{model_key.upper()}** — {pct:.1f}% {bar}")
            lines.append(f"  {description}")
            lines.append(f"  Deviation from ensemble: {delta_str}")

            # Per-model interpretation
            lines.append(f"  → {self._interpret_model_score(model_key, score)}")
            lines.append("")

        return "\n".join(lines)

    def _build_agreement_analysis(self, per_model: dict) -> str:
        valid_models = {k: v for k, v in per_model.items() if v is not None}
        scores = list(valid_models.values())
        if len(scores) < 2:
            return ""

        spread = max(scores) - min(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        mean_score = statistics.mean(scores)

        lines = [
            "🤝 **MODEL AGREEMENT ANALYSIS**",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"• Score spread: {spread * 100:.1f}%",
            f"• Standard deviation: {std_dev * 100:.1f}%",
            f"• Mean confidence: {mean_score * 100:.1f}%",
            "",
        ]

        if spread < 0.10:
            lines.append(
                "✅ **Strong consensus** — All models closely agree, which "
                "significantly increases the reliability of the verdict."
            )
        elif spread < 0.25:
            lines.append(
                "⚡ **Moderate agreement** — Models show some variance, which is "
                "normal. The ensemble weighting accounts for individual model biases."
            )
        else:
            lines.append(
                "⚠️ **Significant disagreement** — Models show notable divergence. "
                "This may indicate subtle or novel manipulation techniques that "
                "affect different analysis dimensions differently."
            )

            # Identify the outlier
            max_model = max(valid_models, key=valid_models.get)  # type: ignore[arg-type]
            min_model = min(valid_models, key=valid_models.get)  # type: ignore[arg-type]
            lines.append(
                f"  Highest alert: **{max_model.upper()}** ({valid_models[max_model]*100:.1f}%)"
            )
            lines.append(
                f"  Lowest alert:  **{min_model.upper()}** ({valid_models[min_model]*100:.1f}%)"
            )

        return "\n".join(lines)

    def _build_temporal_analysis(self, frame_analysis: list[dict]) -> str:
        scores = [f.get("confidence", 0.0) for f in frame_analysis]
        if not scores:
            return ""

        mean_s = statistics.mean(scores)
        std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
        max_s = max(scores)
        min_s = min(scores)
        max_idx = scores.index(max_s)
        min_idx = scores.index(min_s)

        lines = [
            "📈 **TEMPORAL CONSISTENCY ANALYSIS**",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"• Frame count: {len(scores)}",
            f"• Mean confidence: {mean_s * 100:.1f}%",
            f"• Std deviation: {std_s * 100:.1f}%",
            f"• Peak confidence: {max_s * 100:.1f}% (frame {max_idx + 1})",
            f"• Lowest confidence: {min_s * 100:.1f}% (frame {min_idx + 1})",
            "",
        ]

        # Detect spikes (sudden jumps > 15% between consecutive frames)
        spikes: list[int] = []
        for i in range(1, len(scores)):
            if abs(scores[i] - scores[i - 1]) > 0.15:
                spikes.append(i)

        if spikes:
            lines.append(
                f"⚠️  **{len(spikes)} temporal spike(s) detected** at frame(s): "
                f"{', '.join(str(s + 1) for s in spikes)}"
            )
            lines.append(
                "  Sudden confidence jumps often indicate face-swap boundaries, "
                "where the manipulation model transitions between source and target."
            )
        elif std_s > 0.10:
            lines.append(
                "⚡ **Moderate temporal variation** — Frame-to-frame scores fluctuate, "
                "suggesting inconsistent manipulation quality across the sequence."
            )
        else:
            lines.append(
                "✅ **Temporally consistent** — Confidence scores are stable across "
                "frames, indicating either clean authentic footage or a very "
                "sophisticated (uniform) manipulation."
            )

        # Trend detection: rising, falling, or stable
        if len(scores) >= 4:
            first_quarter = statistics.mean(scores[: len(scores) // 4])
            last_quarter = statistics.mean(scores[-(len(scores) // 4) :])
            trend_diff = last_quarter - first_quarter

            if trend_diff > 0.08:
                lines.append(
                    "\n📈 **Rising trend** — Manipulation confidence increases over "
                    "time, possibly indicating degrading deepfake quality in later "
                    "frames or increasing facial movement revealing artifacts."
                )
            elif trend_diff < -0.08:
                lines.append(
                    "\n📉 **Falling trend** — Confidence decreases over time. "
                    "The earlier frames may contain more visible artifacts, or the "
                    "deepfake model adapted better to later frames."
                )

        return "\n".join(lines)

    def _build_spatial_insights(
        self, per_model: dict, confidence: float, verdict: str
    ) -> str:
        lines = [
            "🔬 **SPATIAL ANOMALY INSIGHTS**",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        eff_score = per_model.get("efficientnet", 0.0)
        vit_score = per_model.get("vit", 0.0)

        if verdict == "FAKE":
            if eff_score > 0.7:
                lines.append(
                    "• **Texture artifacts detected** — EfficientNet flagged "
                    "significant pixel-level irregularities, likely including "
                    "blending seams, unnatural skin texture, or compression "
                    "artifacts inconsistent with the rest of the frame."
                )
            if vit_score > 0.7:
                lines.append(
                    "• **Structural inconsistencies found** — The Vision Transformer "
                    "detected global coherence issues such as asymmetric facial "
                    "features, inconsistent lighting direction, or geometric "
                    "distortions in the face region."
                )
            if eff_score > 0.5 and vit_score > 0.5:
                lines.append(
                    "• **Multi-scale artifacts** — Both local (pixel-level) and "
                    "global (structural) anomalies were detected, indicating "
                    "manipulation visible at multiple analysis scales."
                )
        else:
            if eff_score < 0.25 and vit_score < 0.25:
                lines.append(
                    "• **Clean spatial profile** — Neither local texture analysis "
                    "nor global structural analysis detected significant anomalies. "
                    "The face region exhibits natural properties."
                )
            else:
                lines.append(
                    "• **Minor spatial features noted** — Some subtle patterns were "
                    "detected but fall below the manipulation threshold. These may "
                    "be due to compression, lighting conditions, or camera quality."
                )

        return "\n".join(lines)

    def _build_conclusion(
        self, verdict: str, confidence: float, per_model: dict
    ) -> str:
        lines = [
            "📋 **CONCLUSION**",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if verdict == "FAKE" and confidence > self.THRESHOLDS["very_high"]:
            lines.append(
                "This media shows **strong indicators of digital manipulation**. "
                "Multiple independent analysis models concur with high confidence. "
                "The detected artifacts are consistent with known deepfake generation "
                "techniques. Exercise extreme caution when evaluating this content."
            )
        elif verdict == "FAKE" and confidence > self.THRESHOLDS["high"]:
            lines.append(
                "This media **likely contains digital manipulation**. The detection "
                "models identified patterns consistent with face-swapping or "
                "face-reenactment techniques. While confidence is high, independent "
                "verification is recommended for critical decisions."
            )
        elif verdict == "FAKE":
            lines.append(
                "This media shows **moderate indicators of manipulation**. "
                "Some models detected suspicious patterns, but the evidence is "
                "not overwhelming. Further analysis or additional samples may be "
                "needed for a definitive conclusion."
            )
        elif verdict == "REAL" and confidence < self.THRESHOLDS["very_low"]:
            lines.append(
                "This media appears **highly authentic**. All analysis models "
                "report very low manipulation scores, and spatial/temporal "
                "characteristics are consistent with genuine content."
            )
        elif verdict == "REAL":
            lines.append(
                "This media appears **likely authentic**. The analysis models "
                "did not detect significant manipulation artifacts. However, "
                "note that very sophisticated deepfakes may evade detection."
            )
        else:
            lines.append(
                "The analysis is **inconclusive**. The confidence level falls in "
                "an ambiguous range. Consider analysing a higher-quality version "
                "of this media or additional samples from the same source."
            )

        lines.append(
            "\n_Note: AI-based detection is probabilistic. Results should be "
            "interpreted as risk indicators, not definitive proof._"
        )

        return "\n".join(lines)

    # --- Helpers ---------------------------------------------------------- #

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= self.THRESHOLDS["very_high"]:
            return "very high"
        if confidence >= self.THRESHOLDS["high"]:
            return "high"
        if confidence >= self.THRESHOLDS["moderate"]:
            return "moderate"
        if confidence >= self.THRESHOLDS["low"]:
            return "low"
        return "very low"

    @staticmethod
    def _confidence_bar(score: float, length: int = 20) -> str:
        filled = int(round(score * length))
        return f"[{'█' * filled}{'░' * (length - filled)}]"

    def _interpret_model_score(self, model_key: str, score: float) -> str:
        pct = score * 100
        if model_key == "efficientnet":
            if score > 0.7:
                return f"High spatial anomaly score ({pct:.0f}%) — visible blending artifacts or unnatural textures detected."
            if score > 0.4:
                return f"Moderate spatial score ({pct:.0f}%) — some texture irregularities noted, possibly compression-related."
            return f"Low spatial score ({pct:.0f}%) — pixel-level characteristics appear natural."

        if model_key == "vit":
            if score > 0.7:
                return f"High structural anomaly score ({pct:.0f}%) — global face coherence issues detected."
            if score > 0.4:
                return f"Moderate structural score ({pct:.0f}%) — minor asymmetries or lighting inconsistencies."
            return f"Low structural score ({pct:.0f}%) — overall face structure appears consistent."

        if model_key == "cnn_lstm":
            if score > 0.7:
                return f"High temporal anomaly score ({pct:.0f}%) — significant frame-to-frame inconsistencies."
            if score > 0.4:
                return f"Moderate temporal score ({pct:.0f}%) — some inter-frame variations detected."
            return f"Low temporal score ({pct:.0f}%) — temporal dynamics appear natural."

        return f"Score: {pct:.0f}%"

    # ------------------------------------------------------------------ #
    #  Rule-based chat
    # ------------------------------------------------------------------ #

    def _rule_based_chat(self, message: str, context: Optional[dict] = None) -> str:
        """Answer user questions using pattern-matching and context data."""
        msg_lower = message.lower().strip()

        # No context provided
        if not context:
            return (
                "I don't have any detection results to discuss yet. "
                "Please upload a video or image for analysis first, "
                "then I can answer your questions about the results."
            )

        verdict = context.get("verdict", "UNKNOWN")
        confidence = context.get("confidence", 0.0)
        per_model = context.get("per_model_scores", {})
        frame_analysis = context.get("frame_analysis", [])

        # Route to appropriate handler based on question type.
        # Specific topics are checked first so phrases like "explain the
        # model breakdown" or "how confident are you" route to the
        # specialized handler rather than the generic verdict explanation.
        if any(w in msg_lower for w in ["model", "efficientnet", "vit", "cnn", "lstm", "breakdown"]):
            return self._explain_models(per_model, confidence)

        if any(w in msg_lower for w in ["frame", "temporal", "time", "sequence", "spike"]):
            return self._explain_temporal(frame_analysis)

        if any(w in msg_lower for w in ["confidence", "confident", "score", "accurate", "reliable", "trust"]):
            return self._explain_confidence(confidence, per_model)

        if any(w in msg_lower for w in ["artifact", "anomal", "suspicious", "detect"]):
            return self._explain_artifacts(per_model, confidence, verdict)

        if any(w in msg_lower for w in ["why", "how", "explain", "reason", "tell me", "real", "fake", "verdict", "result", "authentic"]):
            return self._explain_verdict(verdict, confidence, per_model, frame_analysis)

        if any(w in msg_lower for w in ["help", "what can", "capability"]):
            return (
                "I'm your AI deepfake investigator. I can help you understand:\n\n"
                "• **Why** the verdict was reached\n"
                "• **Which models** flagged what issues\n"
                "• **Temporal patterns** in video analysis\n"
                "• **Confidence levels** and their meaning\n"
                "• **Specific artifacts** that were detected\n\n"
                "Just ask me anything about the detection results!"
            )

        # Default: provide a summary
        return (
            f"Based on the analysis, this media was classified as **{verdict}** "
            f"with {confidence * 100:.1f}% manipulation confidence.\n\n"
            f"You can ask me about:\n"
            f"• Why this verdict was reached\n"
            f"• How each model contributed\n"
            f"• Temporal frame-by-frame patterns\n"
            f"• What specific artifacts were found"
        )

    def _explain_verdict(
        self,
        verdict: str,
        confidence: float,
        per_model: dict,
        frame_analysis: list[dict],
    ) -> str:
        # Filter out models with no score (e.g. cnn_lstm is None for single images)
        valid_models = {k: v for k, v in per_model.items() if v is not None}
        scores = list(valid_models.values()) or [confidence]

        agreement = (
            "strong consensus"
            if (max(scores) - min(scores)) < 0.10
            else "moderate agreement"
            if (max(scores) - min(scores)) < 0.25
            else "some disagreement"
        )

        response = (
            f"The **{verdict}** verdict was determined by the ensemble of detection "
            f"models, which showed **{agreement}**.\n\n"
        )

        if verdict == "FAKE":
            if valid_models:
                top_model = max(valid_models, key=valid_models.get)  # type: ignore[arg-type]
                response += (
                    f"The strongest signal came from **{top_model.upper()}** "
                    f"({valid_models[top_model]*100:.1f}%), which specializes in "
                    f"{'spatial texture' if top_model == 'efficientnet' else 'structural coherence' if top_model == 'vit' else 'temporal consistency'} "
                    f"analysis.\n\n"
                )
            if len(frame_analysis) > 1:
                scores_list = [f.get("confidence", 0) for f in frame_analysis]
                peak_frame = scores_list.index(max(scores_list)) + 1
                response += (
                    f"Across {len(frame_analysis)} frames, the most suspicious "
                    f"activity was in frame {peak_frame} "
                    f"({max(scores_list)*100:.1f}% confidence)."
                )
        else:
            response += (
                "All models reported low manipulation scores, and no significant "
                "artifacts were detected in the spatial, structural, or temporal "
                "analysis dimensions."
            )

        return response

    def _explain_models(self, per_model: dict, ensemble: float) -> str:
        lines = ["Here's how each detection model contributed:\n"]
        for key, score in per_model.items():
            desc = self.MODEL_DESCRIPTIONS.get(key, key)
            if score is None:
                lines.append(
                    f"• **{key.upper()}** — N/A (not applicable for this input)\n"
                    f"  {desc}\n"
                )
                continue
            delta = score - ensemble
            lines.append(
                f"• **{key.upper()}** — {score*100:.1f}% "
                f"({'↑' if delta > 0 else '↓'}{abs(delta)*100:.1f}% vs ensemble)\n"
                f"  {desc}\n"
            )
        return "\n".join(lines)

    def _explain_temporal(self, frame_analysis: list[dict]) -> str:
        if len(frame_analysis) <= 1:
            return (
                "Temporal analysis is only available for video inputs with "
                "multiple frames. For single images, only spatial analysis applies."
            )

        scores = [f.get("confidence", 0) for f in frame_analysis]
        std_dev = statistics.stdev(scores)

        response = (
            f"**Temporal Analysis** across {len(scores)} frames:\n\n"
            f"• Mean confidence: {statistics.mean(scores)*100:.1f}%\n"
            f"• Variation (std dev): {std_dev*100:.1f}%\n"
            f"• Peak: frame {scores.index(max(scores))+1} ({max(scores)*100:.1f}%)\n"
            f"• Lowest: frame {scores.index(min(scores))+1} ({min(scores)*100:.1f}%)\n\n"
        )

        spikes = [
            i + 1
            for i in range(1, len(scores))
            if abs(scores[i] - scores[i - 1]) > 0.15
        ]
        if spikes:
            response += (
                f"⚠️ Detected {len(spikes)} spike(s) at frame(s) {spikes}. "
                "These sudden jumps may indicate face-swap transition boundaries."
            )
        elif std_dev > 0.10:
            response += (
                "The frame-to-frame variation is moderate, suggesting "
                "inconsistent manipulation quality throughout the video."
            )
        else:
            response += (
                "Scores are stable across frames — the analysis is "
                "temporally consistent."
            )

        return response

    def _explain_confidence(self, confidence: float, per_model: dict) -> str:
        scores = [v for v in per_model.values() if v is not None]
        spread = max(scores) - min(scores) if scores else 0
        label = self._confidence_label(confidence)

        response = (
            f"The overall confidence of **{confidence*100:.1f}%** ({label}) "
            f"is computed by the weighted ensemble of all three models.\n\n"
            f"**Reliability factors:**\n"
            f"• Model agreement (spread: {spread*100:.1f}%): "
            f"{'✅ Strong' if spread < 0.10 else '⚡ Moderate' if spread < 0.25 else '⚠️ Weak'}\n"
        )

        if confidence > 0.9 and spread < 0.10:
            response += (
                "\nThis is a **highly reliable** result — multiple independent "
                "models strongly agree."
            )
        elif confidence < 0.3 and spread < 0.10:
            response += (
                "\nThis is a **highly reliable** authentic assessment — all "
                "models confirm low manipulation scores."
            )
        else:
            response += (
                "\nAs with all AI-based detection, treat this as a risk "
                "indicator rather than absolute proof."
            )

        return response

    def _explain_artifacts(
        self, per_model: dict, confidence: float, verdict: str
    ) -> str:
        if verdict != "FAKE":
            return (
                "No significant manipulation artifacts were detected. "
                "The spatial and structural characteristics of the analyzed "
                "content appear consistent with authentic media."
            )

        artifacts: list[str] = []
        if (per_model.get("efficientnet") or 0) > 0.5:
            artifacts.append(
                "• **Texture anomalies** — Unnatural skin smoothing, blending "
                "boundaries, or inconsistent noise patterns"
            )
        if (per_model.get("vit") or 0) > 0.5:
            artifacts.append(
                "• **Structural issues** — Facial asymmetry beyond natural range, "
                "inconsistent lighting/shadow direction, or geometric warping"
            )
        if (per_model.get("cnn_lstm") or 0) > 0.5:
            artifacts.append(
                "• **Temporal artifacts** — Flickering, inconsistent motion blur, "
                "or unnatural expression transitions between frames"
            )

        if artifacts:
            return "**Detected artifacts:**\n\n" + "\n".join(artifacts)
        return "Some subtle indicators were noted but individual artifact types are below the detail threshold."

    # ------------------------------------------------------------------ #
    #  Gemini-enhanced analysis
    # ------------------------------------------------------------------ #

    def _gemini_enhanced_analysis(
        self, results: dict, rule_analysis: str
    ) -> str:
        """Use Gemini to enrich the rule-based analysis."""
        prompt = (
            "You are an expert forensic media analyst specializing in deepfake detection. "
            "Below is a technical analysis report from our automated detection system. "
            "Please enhance this report with:\n"
            "1. More accessible language for non-technical users\n"
            "2. Additional context about the types of manipulation detected\n"
            "3. Practical recommendations for the user\n"
            "4. Any additional insights based on the detection patterns\n\n"
            "Keep the same structure and key findings, but make it more comprehensive "
            "and easier to understand.\n\n"
            f"---\n\nDetection Data:\n"
            f"Verdict: {results.get('verdict')}\n"
            f"Confidence: {results.get('confidence', 0)*100:.1f}%\n"
            f"Per-model scores: {results.get('per_model_scores')}\n"
            f"Number of frames: {len(results.get('frame_analysis', []))}\n\n"
            f"Automated Analysis:\n{rule_analysis}"
        )

        response = self._gemini_model.generate_content(prompt)
        return response.text

    def _gemini_chat(self, message: str, context: dict) -> str:
        """Use Gemini for conversational responses about detection results."""
        prompt = (
            "You are an expert AI deepfake investigator assistant. "
            "The user is asking about deepfake detection results. "
            "Be informative, precise, and helpful.\n\n"
            f"Detection context:\n"
            f"- Verdict: {context.get('verdict')}\n"
            f"- Confidence: {context.get('confidence', 0)*100:.1f}%\n"
            f"- Per-model scores: {context.get('per_model_scores')}\n"
            f"- Frames analyzed: {len(context.get('frame_analysis', []))}\n\n"
            f"User question: {message}"
        )

        response = self._gemini_model.generate_content(prompt)
        return response.text
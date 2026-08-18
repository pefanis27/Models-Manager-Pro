package com.autonomousdiagnosis.app

import java.util.ArrayDeque
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight motion/IoU tracker for live camera video.
 * Boxes are visible from the first valid detection; final HEALTHY/DISEASE status
 * still requires temporal confirmation.
 */
class VideoDetectionTracker(
    private val historySize: Int = 7,
    private val minDisplayHits: Int = 1,
    private val minDecisionHits: Int = 3,
    private val minAgreement: Float = 0.67f,
    private val decisionConfidence: Float = 0.65f,
    private val maxVisible: Int = 6,
    private val staleMs: Long = 650L,
) {
    private data class Track(
        val id: Int,
        val domain: Int,
        var smooth: Detection,
        var lastSeenMs: Long,
        var hits: Int = 1,
        val history: ArrayDeque<Detection> = ArrayDeque(),
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun clear() {
        tracks.clear()
        nextId = 1
    }

    fun update(detections: List<Detection>, nowMs: Long): List<LiveDetection> {
        val used = mutableSetOf<Int>()
        detections.sortedByDescending { it.confidence }.forEach { detection ->
            val dDomain = DiseaseCatalog.domain(detection.classId)
            var best: Track? = null
            var bestScore = Float.NEGATIVE_INFINITY
            for (track in tracks) {
                if (track.id in used || track.domain != dDomain) continue
                val overlap = iou(track.smooth, detection)
                val motion = normalizedCenterDistance(track.smooth, detection)
                if (overlap < 0.18f && motion > 0.32f) continue
                val score = overlap - motion * 0.15f
                if (score > bestScore) { bestScore = score; best = track }
            }

            val track = best ?: Track(nextId++, dDomain, detection, nowMs).also { tracks += it }
            used += track.id
            if (best != null) {
                track.smooth = smooth(track.smooth, detection, 0.48f)
                track.lastSeenMs = nowMs
                track.hits += 1
            }
            track.history.addLast(detection)
            while (track.history.size > historySize) track.history.removeFirst()
        }

        tracks.removeAll { nowMs - it.lastSeenMs > staleMs }

        return tracks.asSequence()
            .filter { it.hits >= minDisplayHits && nowMs - it.lastSeenMs <= staleMs }
            .map { toLive(it) }
            .sortedWith(compareBy<LiveDetection> {
                when (it.state) {
                    DecisionState.DISEASE -> 0
                    DecisionState.HEALTHY -> 1
                    DecisionState.REVIEW -> 2
                }
            }.thenByDescending { it.detection.confidence })
            .take(maxVisible)
            .toList()
    }

    private fun toLive(track: Track): LiveDetection {
        val votes = IntArray(11)
        val scoreSums = FloatArray(11)
        for (sample in track.history) {
            if (sample.classId in 0..10) {
                votes[sample.classId] += 1
                scoreSums[sample.classId] += sample.confidence
            }
        }
        val winner = (0..10).maxByOrNull { scoreSums[it] } ?: track.smooth.classId
        val winnerCount = votes[winner].coerceAtLeast(1)
        val agreement = winnerCount.toFloat() / track.history.size.coerceAtLeast(1)
        val meanConfidence = scoreSums[winner] / winnerCount
        val confirmed = track.hits >= minDecisionHits && agreement >= minAgreement && meanConfidence >= decisionConfidence
        val state = when {
            !confirmed -> DecisionState.REVIEW
            DiseaseCatalog.isHealthy(winner) -> DecisionState.HEALTHY
            else -> DecisionState.DISEASE
        }
        val shown = track.smooth.copy(confidence = meanConfidence.coerceIn(0f, 1f), classId = winner)
        return LiveDetection(track.id, shown, winner, agreement, track.hits, state)
    }

    private fun smooth(old: Detection, fresh: Detection, alpha: Float): Detection {
        fun mix(a: Float, b: Float) = a * (1f - alpha) + b * alpha
        return fresh.copy(
            x1 = mix(old.x1, fresh.x1), y1 = mix(old.y1, fresh.y1),
            x2 = mix(old.x2, fresh.x2), y2 = mix(old.y2, fresh.y2),
        )
    }

    private fun normalizedCenterDistance(a: Detection, b: Detection): Float {
        val acx = (a.x1 + a.x2) * .5f; val acy = (a.y1 + a.y2) * .5f
        val bcx = (b.x1 + b.x2) * .5f; val bcy = (b.y1 + b.y2) * .5f
        val diag = hypot(max(a.x2-a.x1, b.x2-b.x1), max(a.y2-a.y1, b.y2-b.y1)).coerceAtLeast(1f)
        return hypot(acx-bcx, acy-bcy) / diag
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1=max(a.x1,b.x1); val y1=max(a.y1,b.y1); val x2=min(a.x2,b.x2); val y2=min(a.y2,b.y2)
        val inter=max(0f,x2-x1)*max(0f,y2-y1)
        val aa=max(0f,a.x2-a.x1)*max(0f,a.y2-a.y1); val bb=max(0f,b.x2-b.x1)*max(0f,b.y2-b.y1)
        return inter/(aa+bb-inter).coerceAtLeast(1e-6f)
    }
}

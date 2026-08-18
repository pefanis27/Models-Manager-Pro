package com.autonomousdiagnosis.app

import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.min

class TemporalConsensus(
    private val historySize: Int = 7,
    private val minObservations: Int = 3,
    private val minAgreement: Float = 0.67f,
    private val minMeanConfidence: Float = 0.60f,
) {
    private data class Track(
        val id: Int,
        var box: Detection,
        var missed: Int = 0,
        val history: ArrayDeque<Detection> = ArrayDeque(),
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun clear() {
        tracks.clear()
        nextId = 1
    }

    fun update(detections: List<Detection>): List<LiveDetection> {
        tracks.forEach { it.missed += 1 }
        val usedTracks = mutableSetOf<Int>()
        val output = mutableListOf<LiveDetection>()

        detections.sortedByDescending { it.confidence }.forEach { detection ->
            var best: Track? = null
            var bestIou = 0f
            for (track in tracks) {
                if (track.id in usedTracks) continue
                if (DiseaseCatalog.domain(track.box.classId) != DiseaseCatalog.domain(detection.classId)) continue
                val overlap = iou(track.box, detection)
                if (overlap >= 0.30f && overlap > bestIou) {
                    bestIou = overlap
                    best = track
                }
            }

            val track = best ?: Track(nextId++, detection).also { tracks += it }
            usedTracks += track.id
            track.box = detection
            track.missed = 0
            track.history.addLast(detection)
            while (track.history.size > historySize) track.history.removeFirst()

            val votes = track.history.groupingBy { it.classId }.eachCount()
            val winner = votes.maxByOrNull { it.value }?.key ?: detection.classId
            val winnerCount = votes[winner] ?: 1
            val agreement = winnerCount.toFloat() / track.history.size.coerceAtLeast(1)
            val winnerSamples = track.history.filter { it.classId == winner }
            val meanConfidence = winnerSamples.map { it.confidence }.average().toFloat()
            val confirmed = track.history.size >= minObservations &&
                agreement >= minAgreement && meanConfidence >= minMeanConfidence

            // Suppress single-frame and two-frame flashes. A visible target must first
            // survive the advertised three-frame temporal observation window.
            if (track.history.size < minObservations) return@forEach

            val state = when {
                !confirmed -> DecisionState.REVIEW
                DiseaseCatalog.isHealthy(winner) -> DecisionState.HEALTHY
                else -> DecisionState.DISEASE
            }

            output += LiveDetection(
                detection = detection,
                displayClassId = if (confirmed) winner else detection.classId,
                agreement = agreement,
                observations = track.history.size,
                state = state,
            )
        }

        tracks.removeAll { it.missed > 5 }
        return output
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        return intersection / (areaA + areaB - intersection).coerceAtLeast(1e-6f)
    }
}

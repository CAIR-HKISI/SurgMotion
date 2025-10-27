#!/usr/bin/env python3
"""Test segmental edit distance implementation."""

def compress_segments(sequence):
    """
    Compress consecutive repeated labels into segments.
    Example: [0, 0, 0, 1, 1, 1, 2] -> [0, 1, 2]
             [0, 1, 0, 1, 0] -> [0, 1, 0, 1, 0]
    """
    if len(sequence) == 0:
        return []

    segments = [sequence[0]]
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            segments.append(sequence[i])

    return segments


def levenshtein_distance(seq1, seq2):
    """Calculate Levenshtein (edit) distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def segmental_edit_distance(seq1, seq2):
    """
    Calculate edit distance on compressed segments (not frame-level).
    Returns normalized edit score (0-100 scale).
    """
    # Compress sequences to segments
    segments1 = compress_segments(seq1)
    segments2 = compress_segments(seq2)

    # Calculate edit distance on segments
    edit_dist = levenshtein_distance(segments1, segments2)
    max_len = max(len(segments1), len(segments2))

    if max_len == 0:
        return 0.0

    return (edit_dist / max_len) * 100


# Test cases
print("=" * 70)
print("Testing Segmental Edit Distance Implementation")
print("=" * 70)

# Test 1: Perfect segmentation
gt = [0, 0, 0, 1, 1, 1]
pred = [0, 0, 0, 1, 1, 1]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 1: Perfect segmentation")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Expected: 0.00% (identical segments)")
print(f"  ✓ PASS" if dist == 0.0 else "  ✗ FAIL")

# Test 2: Same segments, different frame counts (should be 0!)
gt = [0, 0, 0, 1, 1, 1]
pred = [0, 0, 1, 1, 1, 1, 1]  # Different frame counts but same segments [0,1]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 2: Same segments, different frame durations")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Expected: 0.00% (identical segments, just different durations)")
print(f"  ✓ PASS" if dist == 0.0 else "  ✗ FAIL")

# Test 3: Extra segment (over-segmentation)
gt = [0, 0, 0, 1, 1, 1]
pred = [0, 0, 0, 1, 0, 1]  # Segments: [0, 1, 0, 1] vs [0, 1]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 3: Over-segmentation (extra segments)")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Expected: 50.00% (2 insertions needed, 2/4 = 50%)")
print(f"  ✓ PASS" if abs(dist - 50.0) < 0.01 else "  ✗ FAIL")

# Test 4: Missing segment (under-segmentation)
gt = [0, 0, 1, 1, 2, 2]
pred = [0, 0, 0, 0, 2, 2]  # Segments: [0, 2] vs [0, 1, 2]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 4: Under-segmentation (missing segment)")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Expected: 33.33% (1 deletion needed, 1/3 = 33.33%)")
print(f"  ✓ PASS" if abs(dist - 33.33) < 0.1 else "  ✗ FAIL")

# Test 5: Wrong segment label
gt = [0, 0, 0, 1, 1, 1, 2, 2, 2]
pred = [0, 0, 0, 3, 3, 3, 2, 2, 2]  # Segments: [0, 3, 2] vs [0, 1, 2]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 5: Wrong segment label (substitution)")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Expected: 33.33% (1 substitution needed, 1/3 = 33.33%)")
print(f"  ✓ PASS" if abs(dist - 33.33) < 0.1 else "  ✗ FAIL")

# Test 6: Realistic surgical case
gt =   [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
pred = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]  # Slightly early phase transitions
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 6: Realistic surgical phase sequence")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Note: Same segments [0,1,2,3], just different frame durations")

# Test 7: Flickering predictions (many segments)
gt =   [0, 0, 0, 0, 1, 1, 1, 1]
pred = [0, 1, 0, 1, 1, 0, 1, 1]  # Segments: [0,1,0,1,1,0,1] vs [0,1]
gt_seg = compress_segments(gt)
pred_seg = compress_segments(pred)
dist = segmental_edit_distance(gt, pred)
print(f"\nTest 7: Flickering predictions (over-segmentation)")
print(f"  GT:         {gt}")
print(f"  Pred:       {pred}")
print(f"  GT seg:     {gt_seg}")
print(f"  Pred seg:   {pred_seg}")
print(f"  Edit Score: {dist:.2f}%")
print(f"  Note: Heavy over-segmentation penalty")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)

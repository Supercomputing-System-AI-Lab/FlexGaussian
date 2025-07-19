import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# -------------------- Bit-Packing Utilities -------------------- #

def pack_4bit_to_8bit(a, b):
    return (a.astype(np.uint8) << 4) | b.astype(np.uint8)

def unpack_8bit_to_4bit(packed_value):
    return (packed_value >> 4) & 0x0F, packed_value & 0x0F

def pack_2bit_to_8bit(a, b, c, d):
    return (
        (a.astype(np.uint8) << 6) |
        (b.astype(np.uint8) << 4) |
        (c.astype(np.uint8) << 2) |
        d.astype(np.uint8)
    )

def unpack_8bit_to_2bit(packed_value):
    return (
        (packed_value >> 6) & 0x03,
        (packed_value >> 4) & 0x03,
        (packed_value >> 2) & 0x03,
        packed_value & 0x03
    )

# -------------------- Int8 Quantization -------------------- #

def asymmetric_quantize_int8_per_segment_parallel(r, num_segments):
    N, K = r.shape
    num_segments = min(max(num_segments, 1), N)
    indices = np.linspace(0, N, num_segments + 1, dtype=int)

    segment_bounds = [
        (indices[i], indices[i + 1])
        for i in range(num_segments)
        if indices[i] < indices[i + 1]
    ]

    scales = np.zeros((len(segment_bounds), K))
    zero_points = np.zeros((len(segment_bounds), K))
    quantized_data = np.zeros_like(r, dtype=np.int8)

    def process_segment(i):
        start, end = segment_bounds[i]
        seg = r[start:end, :]

        alpha = seg.min(axis=0)
        beta = seg.max(axis=0)
        scale = (beta - alpha) / 255.0
        scale[scale == 0] = 1.0
        zero_point = -(alpha / scale + 128)

        q = np.round(seg / scale + zero_point).astype(np.int8)
        q = np.clip(q, -128, 127)
        q[:, scale == 0] = -128

        return i, q, scale, zero_point

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_segment, i) for i in range(len(segment_bounds))]
        for future in futures:
            i, q_seg, scale, zp = future.result()
            start, end = segment_bounds[i]
            quantized_data[start:end, :] = q_seg
            scales[i, :] = scale
            zero_points[i, :] = zp

    return quantized_data, scales, zero_points

def asymmetric_dequantize_int8_per_segment_parallel(quantized_data, scales, zero_points, num_segments, shape):
    N, K = shape
    num_segments = min(num_segments, N)
    indices = np.linspace(0, N, num_segments + 1, dtype=int)

    segment_bounds = [
        (indices[i], indices[i + 1])
        for i in range(num_segments)
        if indices[i] < indices[i + 1]
    ]

    reconstructed = np.zeros(shape, dtype=np.float32)

    def process_segment(i):
        start, end = segment_bounds[i]
        q = quantized_data[start:end, :]
        return i, (q.astype(np.float32) - zero_points[i][np.newaxis, :]) * scales[i][np.newaxis, :]

    with ThreadPoolExecutor() as executor:
        for future in executor.map(process_segment, range(len(segment_bounds))):
            i, seg = future
            start, end = segment_bounds[i]
            reconstructed[start:end, :] = seg

    return reconstructed

# -------------------- Int4 Quantization -------------------- #

def quantize_segment(segment_r, alpha_q=0, beta_q=15):
    alpha = segment_r.min(axis=0)
    beta = segment_r.max(axis=0)
    scale = (beta - alpha) / (beta_q - alpha_q)
    scale[scale == 0] = 1.0
    zero_point = alpha_q - alpha / scale

    quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, alpha_q, beta_q)

    quantized = quantized.reshape(segment_r.shape[0] // 2, 2, segment_r.shape[1])
    packed = pack_4bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :])
    return packed, scale, zero_point

def asymmetric_quantize_int4_segment_parallel(r, num_segments):
    N, K = r.shape
    num_segments = min(num_segments, N)
    segment_size = int(np.ceil(N / num_segments))
    segment_size += segment_size % 2

    total_padded = num_segments * segment_size
    padding = total_padded - N
    r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

    r_segments = r_padded.reshape(num_segments, segment_size, K)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(quantize_segment, r_segments))

    packed_segments, scales, zero_points = zip(*results)
    return (
        np.vstack(packed_segments),
        np.vstack(scales),
        np.vstack(zero_points),
        segment_size,
        N
    )

def dequantize_segment(packed_segment, scale, zero_point, segment_size, K):
    a, b = unpack_8bit_to_4bit(packed_segment)
    quantized = np.empty((segment_size, K), dtype=np.float32)
    quantized[::2], quantized[1::2] = a, b

    scale = scale[np.newaxis, :]
    zero_point = zero_point[np.newaxis, :]
    return (quantized - zero_point) * scale

def asymmetric_dequantize_int4_segment_parallel(packed_matrix, scales, zero_points, segment_size, N_original):
    K = packed_matrix.shape[1]
    segment_size += segment_size % 2
    num_segments = len(scales)
    per_segment = segment_size // 2

    with ThreadPoolExecutor() as executor:
        results = [
            executor.submit(
                dequantize_segment,
                packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                scales[i],
                zero_points[i],
                segment_size,
                K
            )
            for i in range(num_segments)
        ]
        dequantized = [f.result() for f in results]

    return np.vstack(dequantized)[:N_original, :]

# -------------------- Int2 Quantization -------------------- #

def quantize_segment_int2(segment_r, alpha_q=0, beta_q=3):
    alpha = segment_r.min(axis=0)
    beta = segment_r.max(axis=0)
    scale = (beta - alpha) / (beta_q - alpha_q)
    scale[scale == 0] = 1.0
    zero_point = alpha_q - alpha / scale

    quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, alpha_q, beta_q)

    quantized = quantized.reshape(segment_r.shape[0] // 4, 4, segment_r.shape[1])
    packed = pack_2bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :], quantized[:, 2, :], quantized[:, 3, :])
    return packed, scale, zero_point

def asymmetric_quantize_int2_segment_parallel(r, num_segments):
    N, K = r.shape
    num_segments = min(num_segments, N)
    segment_size = int(np.ceil(N / num_segments))
    if segment_size % 4 != 0:
        segment_size += 4 - (segment_size % 4)

    total_padded = num_segments * segment_size
    padding = total_padded - N
    r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

    r_segments = r_padded.reshape(num_segments, segment_size, K)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(quantize_segment_int2, r_segments))

    packed_segments, scales, zero_points = zip(*results)
    return (
        np.vstack(packed_segments),
        np.vstack(scales),
        np.vstack(zero_points),
        segment_size,
        N
    )

def dequantize_segment_int2(packed_segment, scale, zero_point, segment_size, K):
    a, b, c, d = unpack_8bit_to_2bit(packed_segment)
    quantized = np.empty((segment_size, K), dtype=np.float32)
    quantized[::4], quantized[1::4], quantized[2::4], quantized[3::4] = a, b, c, d

    scale = scale[np.newaxis, :]
    zero_point = zero_point[np.newaxis, :]
    return (quantized - zero_point) * scale

def asymmetric_dequantize_int2_segment_parallel(packed_matrix, scales, zero_points, segment_size, N_original):
    K = packed_matrix.shape[1]
    if segment_size % 4 != 0:
        segment_size += 4 - (segment_size % 4)

    per_segment = segment_size // 4
    num_segments = len(scales)

    with ThreadPoolExecutor() as executor:
        results = [
            executor.submit(
                dequantize_segment_int2,
                packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                scales[i],
                zero_points[i],
                segment_size,
                K
            )
            for i in range(num_segments)
        ]
        dequantized = [f.result() for f in results]

    return np.vstack(dequantized)[:N_original, :]
import GalaCore

func zeros(_ shape: [Int], _ dtype: DType) -> CPUStorageBuffer {
    let result = CPUStorageBuffer(shape, dtype)
    switch result.dtype {
        case .float32: result.buffer.withMemoryRebound(to: Float32.self, { $0.initialize(repeating: 0.0) })
        case .int32: result.buffer.withMemoryRebound(to: Int32.self, { $0.initialize(repeating: 0) })
    }
    return result
}

func ones(_ shape: [Int], _ dtype: DType) -> CPUStorageBuffer {
    let result = CPUStorageBuffer(shape, dtype)
    switch result.dtype {
        case .float32: result.buffer.withMemoryRebound(to: Float32.self, { $0.initialize(repeating: 1.0) })
        case .int32: result.buffer.withMemoryRebound(to: Int32.self, { $0.initialize(repeating: 1) })
    }
    return result
}

func fromData(_ shape: [Int], _ dtype: DType, _ data: UnsafeRawBufferPointer) -> CPUStorageBuffer  {
    let result = CPUStorageBuffer(shape, dtype)
    result.buffer.copyMemory(from: data)
    return result
}

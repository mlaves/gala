func zeros(_ storage: CPUStorageBuffer) {
    storage.buffer.withMemoryRebound(to: Float32.self, { $0.initialize(repeating: 0.0) })
}

func ones(_ storage: CPUStorageBuffer) {
    storage.buffer.withMemoryRebound(to: Float32.self, { $0.initialize(repeating: 1.0) })
}

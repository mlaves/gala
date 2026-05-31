import GalaCore

package class CPUStorageBuffer: StorageBuffer {
    package let device: Device = .cpu
    package let shape: [Int]
    package let dtype: DType
    internal let buffer: UnsafeMutableRawBufferPointer

    internal init(_ shape: [Int], _ dtype: DType) {
        self.shape = shape
        self.dtype = dtype
        let byteCount = shape.reduce(1, *) * dtype.byteSize
        buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<Int>.alignment)
    }

    deinit {
        buffer.deallocate()
    }

    internal func typed<T>(as: T.Type) -> UnsafeMutableBufferPointer<T> {
        return buffer.assumingMemoryBound(to: T.self)
    }

    package func cast(to: DType) -> CPUStorageBuffer {
        let from = dtype
        if from == to { return self }
        let result = CPUStorageBuffer(shape, to)

        switch (from, to) {
            case (.int32, .float32):
                let src = self.typed(as: Int32.self)
                let dst = result.typed(as: Float.self)
                precondition(src.count == dst.count)
                for i in 0..<src.count {
                    dst[i] = Float(src[i])
                }
            case (.float32, .int32):
                let src = self.typed(as: Float.self)
                let dst = result.typed(as: Int32.self)
                precondition(src.count == dst.count)
                for i in 0..<src.count {
                    dst[i] = Int32(src[i])
                }
            default: fatalError("cast error: could not cast from \(dtype) to \(to)")
        }
        return result
    }

    package func scalar(at index: Int) -> ScalarValue {
        switch dtype {
            case .float32: return .float32(typed(as: Float.self)[index])
            case .int32: return .int32(typed(as: Int32.self)[index])
        }
    }
}

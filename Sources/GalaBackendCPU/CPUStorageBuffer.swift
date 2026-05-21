import GalaCore

package class CPUStorageBuffer : StorageBuffer {
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

    package func float32(at index: Int) -> Float32 {
        let byteOffset = index * MemoryLayout<Float32>.stride
        return buffer.baseAddress!.advanced(by: byteOffset).withMemoryRebound(to: Float32.self, capacity: 1, { $0.pointee })
    }

    package func copyIn(from: UnsafeBufferPointer<UInt8>) {
        fatalError("Not implemented")
    }

    package func copyOut(to: UnsafeMutableBufferPointer<UInt8>) {
        fatalError("Not implemented")
    }
}

import GalaCore

public class CPUStorageBuffer : StorageBuffer {
    public let device: Device
    public let shape: [Int]
    public let dtype: DType
    let buffer: UnsafeMutableBufferPointer<UInt8>

    internal init(device: Device, shape: [Int], dtype: DType) {
        self.device = device
        self.shape = shape
        self.dtype = dtype
        let byteCount = shape.reduce(1, *) * dtype.byteSize
        buffer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: byteCount)
    }

    deinit {
        buffer.deallocate()
    }

    public func copyIn(from: UnsafeBufferPointer<UInt8>) {
        fatalError("Not implemented")
    }

    public func copyOut(to: UnsafeMutableBufferPointer<UInt8>) {
        fatalError("Not implemented")
    }
}

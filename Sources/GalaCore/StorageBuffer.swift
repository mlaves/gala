public protocol StorageBuffer {
    var device: Device { get }
    var shape: [Int] { get }
    var dtype: DType { get }

    mutating func copyIn(from: UnsafeBufferPointer<UInt8>)
    func copyOut(to: UnsafeMutableBufferPointer<UInt8>)

    func float32(at index: Int) -> Float32
}

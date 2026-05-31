package protocol StorageBuffer {
    var device: Device { get }
    var shape: [Int] { get }
    var dtype: DType { get }

    func scalar(at index: Int) -> ScalarValue
}

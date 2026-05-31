import GalaCore
import GalaDispatch

public struct Tensor {
    public let node: GraphNode  // the computation that produces this
    public var device: Device { node.device }
    public var shape: [Int] { node.shape }
    public var dtype: DType { node.dtype }

    internal init(node: GraphNode) {
        self.node = node
    }

    public static func zeros(shape: [Int], dtype: DType, device: Device) -> Tensor {
        return Tensor(node: GraphNode(op: Op.zeros, device: device, shape: shape, dtype: dtype))
    }

    public static func ones(shape: [Int], dtype: DType, device: Device) -> Tensor {
        return Tensor(node: GraphNode(op: Op.ones, device: device, shape: shape, dtype: dtype))
    }

    public static func fromData(shape: [Int], dtype: DType, device: Device, data: UnsafeRawBufferPointer) -> Tensor {
        let tensor = Tensor(node: GraphNode(op: Op.fromData(ptr: data), device: device, shape: shape, dtype: dtype))
        try! tensor.realize()
        return tensor
    }

    // Realization pulls the tensor out of the lazy graph
    // This is where device dispatch actually fires
    public func realize() throws {
        if node.storage != nil { return }
        let sorted_nodes = Scheduler.topologicalSort(self.node)
        for node in sorted_nodes {
            let buffer = try dispatch(node, node.inputs.map { $0.storage! })
            node.setStorage(buffer)
        }
    }

    subscript(_ indices: Int...) -> ScalarValue? {
        guard indices.count == shape.count else { return nil }
        guard zip(indices, shape).allSatisfy({$0 >= 0 && $0 < $1}) else { return nil }

        var strides = [1]
        for s in shape.reversed().dropLast() {
            strides.append(s*strides.last!)
        }
        strides = strides.reversed()

        try? realize()
        guard let storage = node.storage else { return nil }

        let flatIndex = zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }

        return storage.scalar(at: flatIndex)
    }

    private func add(other: Tensor) -> Tensor {
        precondition(self.shape == other.shape, "shape mismatch: \(self.shape) vs \(other.shape)")
        return Tensor(node: GraphNode(op: Op.add, inputs: [self.node, other.node], device: device, shape: shape, dtype: dtype))
    }

    static func +(left: Tensor, right: Tensor) -> Tensor {
        return left.add(other: right)
    }

    // Autograd hooks live here, but empty now
    public let requiresGrad: Bool = false
    // var gradfn: GradFn? = nil
}

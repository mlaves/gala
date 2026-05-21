import GalaCore
import GalaScheduler

public struct Tensor {
    public let node: GraphNode  // the computation that produces this
    public let device: Device
    public let shape: [Int]
    public let dtype: DType

    internal init(node: GraphNode) {
        self.node = node
        device = node.device
        shape = node.shape
        dtype = node.dtype
    }

    public static func zeros(shape: [Int], dtype: DType, device: Device) -> Tensor {
        return Tensor(node: GraphNode(op: Op.zeros, device: device, shape: shape, dtype: dtype))
    }

    public static func ones(shape: [Int], dtype: DType, device: Device) -> Tensor {
        return Tensor(node: GraphNode(op: Op.ones, device: device, shape: shape, dtype: dtype))
    }

    // Materialization pulls the tensor out of the lazy graph
    // This is where device dispatch actually fires
    public func realize() throws {
        if node.storage != nil { return }
        let sorted_nodes = Scheduler.topologicalSort(self.node)
        for node in sorted_nodes {
            let buffer = try dispatch(node, node.inputs.map { $0.storage! })
            node.setStorage(buffer)
        }
    }

    subscript(_ indices: Int...) -> Float32? {
        guard indices.count == shape.count else { return nil }
        guard zip(indices, shape).allSatisfy({$0 >= 0 && $0 < $1}) else { return nil }

        var strides = [1]
        for s in shape.dropLast() {
            strides.append(s*strides.last!)
        }
        strides = strides.reversed()

        try? realize()
        guard let storage = node.storage else { return nil }

        let flatIndex = zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }

        return storage.float32(at: flatIndex)
    }

    // Autograd hooks live here, but empty now
    public let requiresGrad: Bool = false
    // var gradfn: GradFn? = nil
}

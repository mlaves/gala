import GalaCore

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

    // Materialization pulls the tensor out of the lazy graph
    // This is where device dispatch actually fires
    public func realize() throws -> StorageBuffer {
        fatalError("Not implemented")
    }

    // Autograd hooks live here, but empty now
    public let requiresGrad: Bool = false
    // var gradfn: GradFn? = nil
}

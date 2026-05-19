import Foundation

public enum Device {
    case cpu, metal
}

public enum DType {
    case float32, float16, bfloat16, int32, bool

    public var byteSize: Int {
        switch self {
            case .float32:
                return 4
            case .float16:
                return 2
            case .bfloat16:
                return 2
            case .int32:
                return 4
            case .bool:
                return 1
        }
    }
}

public enum Op {
    // Creation
    case zeros, ones
}

public class GraphNode {
    public let device: Device
    public let shape: [Int]
    public let dtype: DType
    public let id: UUID
    public let op: Op
    public let inputs: [GraphNode]

    public init(op: Op, inputs: [GraphNode] = [], device: Device, shape: [Int], dtype: DType) {
        self.device = device
        self.shape = shape
        self.dtype = dtype
        self.op = op
        self.inputs = inputs
        id = UUID()
    }

    package func setStorage(_ buffer: any StorageBuffer) {
        guard storage == nil else { fatalError("GraphNode storage already realized") }
        self.storage = buffer
    }

    public private(set) var storage: StorageBuffer?

    // Autograd fields
    public var savedForBackward: [GraphNode] = []
    public var gradAccumulator: GraphNode? = nil
}

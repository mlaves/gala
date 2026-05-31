import Foundation

public enum Device {
    case cpu //, metal
}

public enum DType {
    case float32, int32 //, float16, bfloat16 bool

    public var byteSize: Int {
        switch self {
            case .float32:
                return 4
            case .int32:
                return 4
            // case .float16:
            //     return 2
            // case .bfloat16:
            //     return 2
            // case .bool:
            //     return 1
        }
    }

    public static func promote(_ a: DType, _ b: DType) -> DType {
        if a == b { return a }
        return .float32
    }
}

public enum ScalarValue {
    case float32(Float)
    case int32(Int32)

    public var dtype: DType {
        switch self {
        case .float32: return .float32
        case .int32:   return .int32
        }
    }

    public var asFloat: Float {
        switch self {
            case .float32(let x): return x
            case .int32(let x): return Float(x)
        }
    }

    public var asInt32: Int32 {
        switch self {
            case .float32(let x): return Int32(x)
            case .int32(let x): return x
        }
    }
}

public enum Op {
    // Creation
    case zeros, ones
    case fromData(ptr: UnsafeRawBufferPointer)

    // Binary
    case add
}

public class GraphNode {
    public let device: Device
    public let shape: [Int]
    public let dtype: DType
    public let id: UUID
    public let op: Op
    public let inputs: [GraphNode]
    package private(set) var storage: StorageBuffer?

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

    // Autograd fields
    public var savedForBackward: [GraphNode] = []
    public var gradAccumulator: GraphNode? = nil
}

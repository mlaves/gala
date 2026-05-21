import GalaBackendCPU
import GalaCore

func dispatch(_ node: GraphNode, _ inputs: [any StorageBuffer]) throws -> any StorageBuffer {
    switch node.device {
        case .cpu: try CPUBackendExecutor.execute(node, inputs)
    }
}

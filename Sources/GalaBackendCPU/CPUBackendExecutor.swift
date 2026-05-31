import GalaCore

package class CPUBackendExecutor : BackendExecutor {
    package static func execute(_ node: GraphNode, _ inputs: [StorageBuffer]) throws -> StorageBuffer {
        if !supports(node.op) { throw BackendExecutorError.opNotSupported }
        for i in inputs.dropFirst() {
            if i.dtype != inputs.first!.dtype { throw BackendExecutorError.deviceMismatch }
        }

        switch node.op {
            case .zeros:
                return zeros(node.shape, node.dtype)
            case .ones:
                return ones(node.shape, node.dtype)
            case .fromData(let data):
                return fromData(node.shape, node.dtype, data)
            case .add:
                return add(inputs as! [CPUStorageBuffer])
        }
    }

    package static func supports(_ op: Op) -> Bool {
        switch op {
            case .zeros: return true
            case .ones: return true
            case .fromData(_): return true
            case .add: return true
        }
    }

    package static func synchronize() {
        fatalError("Not implemented")
    }
}

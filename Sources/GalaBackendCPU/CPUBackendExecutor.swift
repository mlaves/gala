import GalaCore

package class CPUBackendExecutor : BackendExecutor {
    package static func execute(_ node: GraphNode, _ inputs: [StorageBuffer]) throws -> StorageBuffer {
        if !supports(node.op) { throw BackendExecutorError.opNotSupported }

        let storage = allocate(node.shape, node.dtype)

        switch node.op {
            case .zeros: zeros(storage as! CPUStorageBuffer)
            case .ones: ones(storage as! CPUStorageBuffer)
        }

        return storage
    }

    package static func supports(_ op: Op) -> Bool {
        switch op {
            case .zeros: return true
            case .ones: return true
        }
    }

    package static func allocate(_ shape: [Int], _ dtype: DType) -> StorageBuffer {
        return CPUStorageBuffer(shape, dtype)
    }

    package static func synchronize() {
        fatalError("Not implemented")
    }
}

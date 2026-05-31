public enum BackendExecutorError: Error {
    case opNotSupported, deviceMismatch
}

package protocol BackendExecutor {
    static func execute(_ node: GraphNode, _ inputs: [StorageBuffer]) throws -> StorageBuffer
    static func supports(_ op: Op) -> Bool
    static func synchronize()
}

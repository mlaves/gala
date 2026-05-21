public enum BackendExecutorError: Error {
    case opNotSupported
}

package protocol BackendExecutor {
    static func execute(_ node: GraphNode, _ inputs: [StorageBuffer]) throws -> StorageBuffer
    static func supports(_ op: Op) -> Bool
    static func allocate(_ shape: [Int], _ dtype: DType) -> StorageBuffer
    static func synchronize()
}

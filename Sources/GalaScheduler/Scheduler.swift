import GalaCore
import Foundation

package struct Scheduler {
    package static func topologicalSort(_ rootNode: GraphNode) -> [GraphNode] {
        var visited: Set<UUID> = []
        var result: [GraphNode] = []
        func dfs(_ currentNode: GraphNode, _ visited: inout Set<UUID>, _ result: inout [GraphNode]) {
            fatalError("Not implemented")
        }
        dfs(rootNode, &visited, &result)
        return result
    }
}

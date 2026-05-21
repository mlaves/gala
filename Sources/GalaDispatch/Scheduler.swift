import GalaCore
import Foundation

package struct Scheduler {
    static package func topologicalSort(_ rootNode: GraphNode) -> [GraphNode] {
        var visited: Set<UUID> = []
        var result: [GraphNode] = []
        func dfs(_ currentNode: GraphNode, _ visited: inout Set<UUID>, _ result: inout [GraphNode]) {
            if currentNode.storage != nil { return }  // already realized
            if visited.contains(currentNode.id) { return }
            visited.insert(currentNode.id)
            for input in currentNode.inputs {
                dfs(input, &visited, &result)
            }
            result.append(currentNode)
        }
        dfs(rootNode, &visited, &result)
        return result
    }
}

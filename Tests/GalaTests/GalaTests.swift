import Testing
@testable import Gala

@Test func example() async throws {
   let tensor = Tensor.zeros(shape: [1, 1, 1], dtype: .float32, device: .cpu)
   let tensor_real = tensor.realize()
   print(tensor_real)
}

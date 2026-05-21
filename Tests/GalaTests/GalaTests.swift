import Testing
@testable import Gala

@Suite struct CPUBackendTests {
    @Test func zerosCPU() throws {
        let tensor = Tensor.zeros(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        try tensor.realize()
        #expect(tensor[0,0,0,0] == Float32(0.0))
        #expect(tensor[1,2,3,4] == Float32(0.0))
        #expect(tensor[0,0,0] == nil)
        #expect(tensor[2,3,4,5] == nil)

        let tensor2 = Tensor.zeros(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        // implicit realization
        #expect(tensor2[0,0,0,0] == Float32(0.0))
        #expect(tensor2[1,2,3,4] == Float32(0.0))
        #expect(tensor2[0,0,0] == nil)
        #expect(tensor2[2,3,4,5] == nil)
    }

    @Test func onesCPU() throws {
        let tensor = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        try tensor.realize()
        #expect(tensor[0,0,0,0] == Float32(1.0))
        #expect(tensor[1,2,3,4] == Float32(1.0))
        #expect(tensor[0,0,0] == nil)
        #expect(tensor[2,3,4,5] == nil)

        let tensor2 = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        // implicit realization
        #expect(tensor2[0,0,0,0] == Float32(1.0))
        #expect(tensor2[1,2,3,4] == Float32(1.0))
        #expect(tensor2[0,0,0] == nil)
        #expect(tensor2[2,3,4,5] == nil)
    }
}

import Testing
@testable import Gala

@Suite struct CPUBackendTests {
    @Test func zerosCPU() throws {
        let tensor = Tensor.zeros(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        try tensor.realize()
        #expect(tensor[0,0,0,0]?.dtype == .float32)
        #expect(tensor[0,0,0,0]?.asFloat == 0.0)
        #expect(tensor[1,2,3,4]?.asFloat == 0.0)
        #expect(tensor[0,0,0] == nil)
        #expect(tensor[2,3,4,5] == nil)

        let tensor2 = Tensor.zeros(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        // implicit realization
        #expect(tensor2[0,0,0,0]?.dtype == .float32)
        #expect(tensor2[0,0,0,0]?.asFloat == 0.0)
        #expect(tensor2[1,2,3,4]?.asFloat == 0.0)
        #expect(tensor2[0,0,0] == nil)
        #expect(tensor2[2,3,4,5] == nil)
    }

    @Test func onesCPU() throws {
        let tensor = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        try tensor.realize()
        #expect(tensor[0,0,0,0]?.dtype == .float32)
        #expect(tensor[0,0,0,0]?.asFloat == 1.0)
        #expect(tensor[1,2,3,4]?.asFloat == 1.0)
        #expect(tensor[0,0,0] == nil)
        #expect(tensor[2,3,4,5] == nil)

        let tensor2 = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        // implicit realization
        #expect(tensor2[0,0,0,0]?.dtype == .float32)
        #expect(tensor2[0,0,0,0]?.asFloat == 1.0)
        #expect(tensor2[1,2,3,4]?.asFloat == 1.0)
        #expect(tensor2[0,0,0] == nil)
        #expect(tensor2[2,3,4,5] == nil)
    }

    @Test func addCPU() throws {
        let lhs = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        let rhs = Tensor.ones(shape: [2, 3, 4, 5], dtype: .float32, device: .cpu)
        let out = lhs + rhs
        try out.realize()
        #expect(out[0,0,0,0]?.asFloat == 2.0)
        #expect(out[0,0,0,0]?.dtype == .float32)
    }

    @Test func fromdataCPU() throws {
        let shape = [1,2,3,4]
        for value: Float in [1.0, -1.0, 13.37] {
            let data = [Float32](repeating: value, count: shape.reduce(1, *))
            data.withUnsafeBufferPointer({ ptr in
                let fromData = Tensor.fromData(shape: shape, dtype: .float32, device: .cpu, data: UnsafeRawBufferPointer(ptr))
                #expect(fromData[0,0,0,0]?.asFloat == value)
                #expect(fromData[0,1,1,1]?.asFloat == value)
                #expect(fromData[0,0,0,0]?.dtype == .float32)
            })
        }
    }
}

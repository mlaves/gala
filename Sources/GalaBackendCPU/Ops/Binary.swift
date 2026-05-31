import GalaCore

func add(_ inputs: [CPUStorageBuffer]) -> CPUStorageBuffer {
    precondition(inputs.count == 2)
    let outDType = DType.promote(inputs[0].dtype, inputs[1].dtype)
    let lhs = inputs[0].cast(to: outDType)
    let rhs = inputs[1].cast(to: outDType)
    let result = CPUStorageBuffer(inputs[0].shape, outDType)

    // dispatch
    switch outDType {
        case .float32:
            var result_typed = result.typed(as: Float.self)
            vAdd(lhs.typed(as: Float.self), rhs.typed(as: Float.self), &result_typed)
        case .int32:
            var result_typed = result.typed(as: Int32.self)
            vAdd(lhs.typed(as: Int32.self), rhs.typed(as: Int32.self), &result_typed)
    }
    return result
}

private func vAdd<T: BinaryFloatingPoint>(
    _ left: UnsafeMutableBufferPointer<T>,
    _ right: UnsafeMutableBufferPointer<T>,
    _ out: inout UnsafeMutableBufferPointer<T>
) {
    for i in 0..<left.count {
        out[i] = left[i] + right[i]
    }
}

private func vAdd<T: FixedWidthInteger>(
        _ left: UnsafeMutableBufferPointer<T>,
        _ right: UnsafeMutableBufferPointer<T>,
        _ out: inout UnsafeMutableBufferPointer<T>
    ) {
    precondition(left.count == right.count && left.count == out.count)
    for i in 0..<left.count {
        out[i] = left[i] &+ right[i]
    }
}

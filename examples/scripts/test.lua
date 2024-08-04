function onAttach()
    print("onAttach")
    local vec = Vec4:new(0.0, 1.0, 2.0, 3.0)
    print("vec.x = " .. vec.x)
    vec.x = 0.5
    print("vec.x = " .. vec.x)
    vec = vec * 2.0
    print("vec.x = " .. vec.x)
    vec = vec / 2.0
    print("vec.x = " .. vec.x)
    local vec2 = Vec4:new(1.0, 0.0, 0.0, 0.0)
    vec = vec + vec2
    print("vec.x = " .. vec.x)
    local mat = Mat2:new()
    print("mat(0, 1) = " .. mat(0, 1))
    mat:set(0, 1, vec.x)
    print("mat(0, 1) = " .. mat(0,1))
    i = 0
end

function onUpdate()
    -- print("onUpdate")
    i = i + 1
    seed:set((UVec2:new(math.random(0, 255), math.random(0, 255))))
    update()
    return i < 32
end

function onDetach()
    print("onDetach")
end

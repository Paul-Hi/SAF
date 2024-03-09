function onAttach()
    print("onAttach")
    i = 0
end
function onUpdate()
    print("onUpdate")
    i = i + 1
    return i < 10
end
function onDetach()
    print("onDetach")
end

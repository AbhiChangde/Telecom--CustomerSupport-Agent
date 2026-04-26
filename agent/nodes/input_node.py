from agent.state import SupportAgentState


async def input_node(state: SupportAgentState) -> SupportAgentState:
    # user_input and input_mode are already set by the API route before graph invocation.
    # This node is a pass-through; extend here for input validation or normalisation.
    return state

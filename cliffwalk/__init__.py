from gym.envs.registration import register

register(
    id = "CliffWalk-v0",
    entry_point = "cliffwalk.env:CliffWalk",
)

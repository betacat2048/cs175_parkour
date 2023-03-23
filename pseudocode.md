```python
# inital
replay_memory = RingBuffer()
policy_net_a, policy_net_b = DQN(), DQN()
target_net_a, target_net_b = policy_net_a, policy_net_b

while not converge:
    s = get_state()
    Qs = policy_net_a(s) if rand() > 0.5 else policy_net_b(s)
    # ε-greedy exploring (epsilon & theta decrease during training)
    if rand() < epsilon:
        Qs += rand_normal(std(Qs) * theta)
    # take action and get result
    a = argmax(Qs)
    s_, r = take_action(a)
    # save into the FIFO buffer
    replay_memory.append(s, a, r, s_)

    # training DNN
    batch = replay_memory.sample(batch_size)
    for s, a, r, s_ in batch:
        if rand() < 0.5:
            # learning net a
            a_ = argmax(target_net_a(s_))
            policy_net_a(s)[a] <== r + gamma * min(target_net_a(s_)[a_], target_net_b(s_)[a_])
        else:
            # learning net b
            a_ = argmax(target_net_b(s_))
            policy_net_b(s)[a] <== r + gamma * min(target_net_a(s_)[a_], target_net_b(s_)[a_])

        # soft-update step
        target_net_a = (1 - tau) * target_net_a + tau * policy_net_a
        target_net_b = (1 - tau) * target_net_b + tau * policy_net_b
```


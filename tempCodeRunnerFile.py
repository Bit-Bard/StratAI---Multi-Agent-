        state_tensor = torch.FloatTensor(state)
        probs_B, _ = model(state_tensor)
        action2 = torch.argmax(probs_B).item()
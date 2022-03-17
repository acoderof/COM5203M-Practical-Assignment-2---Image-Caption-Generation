# Train the models
total_step = len(train_loader)
for epoch in range(num_epochs):
    decoder.train()
    for i, (features, captions, lengths) in enumerate(train_loader):

        captions = captions.to(device)
        lens=lengths.squeeze(1)

        targets = pack_padded_sequence(captions, lens, batch_first=True,enforce_sorted=False)[0]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Forward, backward and optimize
            outputs = decoder(features, captions, lengths)
  
            loss = criterion(outputs, targets)          

            loss.backward()
            optimizer.step()

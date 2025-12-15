Version 2 adds support in the hardware library for GATConv, GCNConv, SAGEConv, SAGEGAT and Linear layers. 

For example, check the file demo_sgrace.pynb and GAT_PYNQ module to see how different layers can be used:


self.conv22 = GCNConv_SGRACE(hidden_channels*head_count, hidden_channels)

or

self.conv22 = SAGEConv_SGRACE(hidden_channels*head_count, hidden_channels)

or

self.conv22 = GATConv_SGRACE(hidden_channels*head_count, hidden_channels,1)

or

self.conv22 = SAGEGAT_SGRACE(hidden_channels*head_count, hidden_channels,1)

or

self.lin = Linear_SGRACE(hidden_channels, dataset.num_classes)

To use these layers in emulation mode without an FPGA set acc = 0 #use accelerator in forward path in config.py


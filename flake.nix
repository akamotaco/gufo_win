{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:franzmoca/gufo/a8844cc6977b4c52349b1622f1b3f02e31fead25";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}

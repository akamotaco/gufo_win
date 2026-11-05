{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:franzmoca/gufo/a8774bbc4b8231d49fa772fc0eb7690672267653";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}

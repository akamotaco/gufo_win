{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:gufo-org/gufo/d2b01f24a9a70104170bfe69bd4752fb7ce8e4e1";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}

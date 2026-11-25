import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            Tab("Run", systemImage: "camera") {
                RunView()
            }
            Tab("Plan", systemImage: "cpu") {
                ComputePlanView()
            }
        }
    }
}

#Preview {
    ContentView()
}

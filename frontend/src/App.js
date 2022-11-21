import Header from './components/header';
import Home from './components/home';
import Footer from './components/footer';
import './App.css';

function App() {
  return (
    <div className="App min-h-screen flex flex-col justify-center items-center">
      <Header />
      <Home />
      <Footer />
    </div>
  );
}

export default App;

import Header from './components/header';
import Home from './components/home';
import Footer from './components/footer';
import './App.css';

function App() {
  return (
    <div className="App bg-[#fff] p-[30px] m-[15px] rounded-[30px] shadow-[rgba(149,157,165,0.2)_0px_8px_24px]">
      <Header />
      <Home />
      <Footer />
    </div>
  );
}

export default App;

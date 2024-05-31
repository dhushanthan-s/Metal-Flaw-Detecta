import React from 'react';
import { Link, BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import Predict from './Predict';
import About from './About';

function Navbar() {
    return (
        <Router>
            <div className="navbar">
                <ul>
                    <li><Link to="/">Home</Link></li>
                    <li><Link to="/predict">Predict</Link></li>
                    <li><Link to="/about">About</Link></li>
                </ul>
            </div>
            <Routes>
                <Route exact path="/" component={Home} />
                <Route path="/predict" component={Predict} />
                <Route path="/about" component={About} />
            </Routes>
        </Router>
    );
}

export default Navbar;
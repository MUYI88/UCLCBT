Notepad++ v7.7.1 enhancements and bug-fixes:
pragma solidity ^0.4.24 ;

contract owned {
    address public owner;

    constructor () public {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    function transferOwnerShip(address newOwner) public onlyOwner {
        owner = newOwner;
    }

}

